import asyncio
import json
import os

import uvicorn
import websockets
from app.configs import get_config_instance
from fastapi import APIRouter, WebSocket,FastAPI
# from fastapi.websockets import
import time
import logging
import tracemalloc
import numpy as np
import argparse
import ssl
from modelscope import snapshot_download
from funasr import AutoModel

funasr_router = FastAPI()
base_config = get_config_instance()

# 从 config.ini 中读取基本的配置信息
proj_dir = base_config.proj_dir
asr_model_path = base_config.asr_model_path
local_dir = os.path.join(proj_dir, asr_model_path)

asr_model = base_config.ali_asr_model
asr_model_online = base_config.ali_asr_online_model
vad_model = base_config.ali_vad_model
punc_model = base_config.ali_punc_model
ali_model_revision = base_config.ali_model_revision
ngpu = int(base_config.ngpu)
ncpu = int(base_config.ncpu)
device = base_config.device


# 设置下载地址
asr_model = snapshot_download(asr_model, cache_dir=local_dir)
asr_model_online = snapshot_download(asr_model_online, cache_dir=local_dir)
vad_model = snapshot_download(vad_model, cache_dir=local_dir)
punc_model = snapshot_download(punc_model, cache_dir=local_dir)



print("model loading")

# asr
model_asr = AutoModel(model=asr_model,
                      model_revision=ali_model_revision,
                      ngpu=ngpu,
                      ncpu=ncpu,
                      device=device,
                      disable_pbar=True,
                      disable_log=True,
                      )
# asr
model_asr_streaming = AutoModel(model=asr_model_online,
                                model_revision=ali_model_revision,
                                ngpu=ngpu,
                                ncpu=ncpu,
                                device=device,
                                disable_pbar=True,
                                disable_log=True,
                                )
# vad
model_vad = AutoModel(model=vad_model,
                      model_revision=ali_model_revision,
                      ngpu=ngpu,
                      ncpu=ncpu,
                      device=device,
                      disable_pbar=True,
                      disable_log=True,
                      # chunk_size=60,
                      )
# punc
model_punc = AutoModel(model=punc_model,
                       model_revision=ali_model_revision,
                       ngpu=ngpu,
                       ncpu=ncpu,
                       device=device,
                       disable_pbar=True,
                       disable_log=True,
                       )

print("model loaded! only support one client at the same time now!!!!")


async def async_vad(websocket, audio_in):
    segments_result = model_vad.generate(input=audio_in, **websocket.status_dict_vad)[0]["value"]
    # print(segments_result)

    speech_start = -1
    speech_end = -1

    if len(segments_result) == 0 or len(segments_result) > 1:
        return speech_start, speech_end
    if segments_result[0][0] != -1:
        speech_start = segments_result[0][0]
    if segments_result[0][1] != -1:
        speech_end = segments_result[0][1]
    return speech_start, speech_end


async def async_asr(websocket, audio_in):
    if len(audio_in) > 0:
        # print(len(audio_in))
        print("开始进行asr_offline...")
        rec_result = model_asr.generate(input=audio_in, **websocket.status_dict_asr)[0]
        # print("offline_asr, ", rec_result)
        if model_punc is not None and len(rec_result["text"]) > 0:
            # print("offline, before punc", rec_result, "cache", websocket.status_dict_punc)
            rec_result = model_punc.generate(input=rec_result['text'], **websocket.status_dict_punc)[0]
        # print("offline, after punc", rec_result)
        if len(rec_result["text"]) > 0:
            # print("offline", rec_result)
            mode = "2pass-offline" if "2pass" in websocket.mode else websocket.mode
            print(f"asr_offline : {rec_result['text']}")
            message = {"mode": mode, "text": rec_result["text"], "wav_name": websocket.wav_name,
                                  "is_final": websocket.is_speaking}
            await websocket.send_json(message)


async def async_asr_online(websocket, audio_in):
    if len(audio_in) > 0:
        audio_online_list.append(audio_in)
        print("开始进行asr_online...")
        # print(websocket.status_dict_asr_online.get("is_final", False))
        rec_result = model_asr_streaming.generate(input=audio_in, **websocket.status_dict_asr_online)[0]
        print("online, ", rec_result)
        if websocket.mode == "2pass" and websocket.status_dict_asr_online.get("is_final", False):
            return
        #     websocket.status_dict_asr_online["cache"] = dict()
        if len(rec_result["text"]):
            mode = "2pass-online" if "2pass" in websocket.mode else websocket.mode
            print(f"asr_online : {rec_result['text']}")
            message = {"mode": mode, "text": rec_result["text"], "wav_name": websocket.wav_name,
                                  "is_final": websocket.is_speaking}
            await websocket.send_json(message)

# start_server = websockets.serve(ws_serve, args.host, args.port, subprotocols=["binary"], ping_interval=None)
# asyncio.get_event_loop().run_until_complete(start_server)
# asyncio.get_event_loop().run_forever()

import wave
# 音频参数
sample_width = 2  # 样本宽度（以字节为单位）
sample_rate = 16000  # 采样率（每秒采样次数）
channels = 1  # 声道数（单声道）
audio_list = []
audio_online_list = []
def bytes_to_wav(byte_data, sample_width, sample_rate, channels,filename:str):
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(byte_data)

@funasr_router.websocket("/recognition_wss/ali")
async def ws_serve(websocket: WebSocket):
    await websocket.accept()
    receive_counts = 0
    frames = []
    frames_asr = []
    frames_asr_online = []
    websocket.status_dict_asr = {}
    websocket.status_dict_asr_online = {"cache": {}, "is_final": False}
    websocket.status_dict_vad = {'cache': {}, "is_final": False}
    websocket.status_dict_punc = {'cache': {}}
    websocket.chunk_interval = 10
    websocket.vad_pre_idx = 0
    speech_start = False
    speech_end_i = -1
    websocket.wav_name = "microphone"
    websocket.mode = "2pass"
    print("new user connected")
    while True:
        message = await websocket.receive()
        message_type = message["type"]
        # 根据接受到的数据类型，对接收信息进行解析
        if message_type == "websocket.disconnect":
            print("ConnectionClosed...")
            byte_array = bytes().join(audio_list)
            byte_online_array = bytes().join(audio_online_list)
            bytes_to_wav(byte_array, sample_width, sample_rate, channels,"audio.wav")
            bytes_to_wav(byte_array, sample_width, sample_rate, channels, "online_audio.wav")
            audio_list.clear()
            break
        elif message_type == "websocket.receive":
            # 字节信息为 音频数据， 文本信息为 初始化信息
            message = message.get("bytes") or message.get("text")
            receive_counts = receive_counts + 1
            # 因为 C# Websocket 只能发送 bytes 格式的数据，所以要判断是否为json格式，并且包含 is_speaking
            if isinstance(message,bytes):
                try:
                    # 如果可以按照utf-8编码解码，说明是 json string
                    message = message.decode('utf-8')
                    print("解码成功！！！",message)
                    # print(f"[msg-{receive_counts}] 接收到config数据...")
                except:
                    audio_list.append(message)
                    # 如果不可以，说明是音频数据
                    # print(f"[msg-{receive_counts}] 接收到音频数据...")
                    pass

        # 判断收到的是不是第一帧，用于初始化音频处理信息
        if isinstance(message,str):
            messagejson = json.loads(message)
            print('接收到第一帧！帧内容为：',messagejson)
            if "is_speaking" in messagejson:
                websocket.is_speaking = messagejson["is_speaking"]
                websocket.status_dict_asr_online["is_final"] = not websocket.is_speaking
            if "chunk_interval" in messagejson:
                websocket.chunk_interval = messagejson["chunk_interval"]
            if "wav_name" in messagejson:
                websocket.wav_name = messagejson.get("wav_name")
            if "chunk_size" in messagejson:
                # print("成功获得到chunk_size...")
                chunk_size = messagejson["chunk_size"]
                if isinstance(chunk_size, str):
                    chunk_size = chunk_size.split(',')
                websocket.status_dict_asr_online["chunk_size"] = [int(x) for x in chunk_size]
            if "encoder_chunk_look_back" in messagejson:
                websocket.status_dict_asr_online["encoder_chunk_look_back"] = messagejson["encoder_chunk_look_back"]
            if "decoder_chunk_look_back" in messagejson:
                websocket.status_dict_asr_online["decoder_chunk_look_back"] = messagejson["decoder_chunk_look_back"]
            if "hotword" in messagejson:
                websocket.status_dict_asr["hotword"] = messagejson["hotword"]
            if "mode" in messagejson:
                # 实时语音听写服务（online）
                # 非实时一句话转写（offline）
                # 实时与非实时一体化协同（2pass）
                websocket.mode = messagejson["mode"]
        websocket.status_dict_vad["chunk_size"] = int(
            websocket.status_dict_asr_online["chunk_size"][1] * 60 / websocket.chunk_interval)
        # 如果保存的音频帧超过了0 或者 message 不再是json string格式，说明当前message是音频帧，应该保存
        if len(frames_asr_online) > 0 or len(frames_asr) > 0 or not isinstance(message, str):
            # 如果message不是str，则说明是音频数据
            if not isinstance(message, str):
                frames.append(message)
                duration_ms = len(message) // 32
                websocket.vad_pre_idx += duration_ms

                # asr online
                frames_asr_online.append(message)
                # 如果是最后一帧，那么speech_end_i ！= -1
                # 如果是普通音频帧，那么speech_end_i == -1
                websocket.status_dict_asr_online["is_final"] = speech_end_i != -1
                if len(frames_asr_online) % websocket.chunk_interval == 0 or websocket.status_dict_asr_online[
                    "is_final"]:
                    if websocket.mode == "2pass" or websocket.mode == "online":
                        audio_in = b"".join(frames_asr_online)
                        # try:

                        await async_asr_online(websocket, audio_in)
                        # except:
                        #     print(f"error in asr streaming, {websocket.status_dict_asr_online}")
                    frames_asr_online = []
                if speech_start:
                    frames_asr.append(message)
                # vad online
                speech_start_i, speech_end_i = await async_vad(websocket, message)
                # 如果 speech_start_i ！= -1，说明vad检测到了音频活动，所以 speech_start_i 一定不等于-1
                # 但是 speech_end_i 不一定等于 -1
                if speech_start_i != -1:
                    speech_start = True
                    beg_bias = (websocket.vad_pre_idx - speech_start_i) // duration_ms
                    frames_pre = frames[-beg_bias:]
                    frames_asr = []
                    frames_asr.extend(frames_pre)
            # asr punc offline
            # 说明 vad 检测到人声活动结束
            if speech_end_i != -1 or not websocket.is_speaking:
                # print("vad end point")
                if websocket.mode == "2pass" or websocket.mode == "offline":
                    audio_in = b"".join(frames_asr)
                    # try:

                    await async_asr(websocket, audio_in)
                    # except:
                    #     print("error in asr offline")
                frames_asr = []
                speech_start = False
                frames_asr_online = []
                websocket.status_dict_asr_online["cache"] = {}
                if not websocket.is_speaking:
                    websocket.vad_pre_idx = 0
                    frames = []
                    websocket.status_dict_vad["cache"] = {}
                else:
                    frames = frames[-20:]

if __name__ == '__main__':
    host = base_config.ipaddress
    port = int(base_config.asr_port)
    uvicorn.run(funasr_router,host=host,port=port)
    byte_array = bytes().join(audio_list)
    bytes_to_wav(byte_array, sample_width, sample_rate, channels)
    print("结束了！")