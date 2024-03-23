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


websocket_users = set()

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


async def ws_reset(websocket):
    print("ws reset now, total num is ", len(websocket_users))

    websocket.status_dict_asr_online["cache"] = {}
    websocket.status_dict_asr_online["is_final"] = True
    websocket.status_dict_vad["cache"] = {}
    websocket.status_dict_vad["is_final"] = True
    websocket.status_dict_punc["cache"] = {}

    await websocket.close()


async def clear_websocket():
    for websocket in websocket_users:
        await ws_reset(websocket)
    websocket_users.clear()


# async def ws_serve(websocket, path):
#     frames = []
#     frames_asr = []
#     frames_asr_online = []
#     global websocket_users
#     # await clear_websocket()
#     websocket_users.add(websocket)
#     websocket.status_dict_asr = {}
#     websocket.status_dict_asr_online = {"cache": {}, "is_final": False}
#     websocket.status_dict_vad = {'cache': {}, "is_final": False}
#     websocket.status_dict_punc = {'cache': {}}
#     websocket.chunk_interval = 10
#     websocket.vad_pre_idx = 0
#     speech_start = False
#     speech_end_i = -1
#     websocket.wav_name = "microphone"
#     websocket.mode = "2pass"
#     print("new user connected", flush=True)
#
#     try:
#         async for message in websocket:
#             if isinstance(message, str):
#                 messagejson = json.loads(message)
#
#                 if "is_speaking" in messagejson:
#                     websocket.is_speaking = messagejson["is_speaking"]
#                     websocket.status_dict_asr_online["is_final"] = not websocket.is_speaking
#                 if "chunk_interval" in messagejson:
#                     websocket.chunk_interval = messagejson["chunk_interval"]
#                 if "wav_name" in messagejson:
#                     websocket.wav_name = messagejson.get("wav_name")
#                 if "chunk_size" in messagejson:
#                     chunk_size = messagejson["chunk_size"]
#                     if isinstance(chunk_size, str):
#                         chunk_size = chunk_size.split(',')
#                     websocket.status_dict_asr_online["chunk_size"] = [int(x) for x in chunk_size]
#                 if "encoder_chunk_look_back" in messagejson:
#                     websocket.status_dict_asr_online["encoder_chunk_look_back"] = messagejson["encoder_chunk_look_back"]
#                 if "decoder_chunk_look_back" in messagejson:
#                     websocket.status_dict_asr_online["decoder_chunk_look_back"] = messagejson["decoder_chunk_look_back"]
#                 if "hotword" in messagejson:
#                     websocket.status_dict_asr["hotword"] = messagejson["hotword"]
#                 if "mode" in messagejson:
#                     websocket.mode = messagejson["mode"]
#
#             websocket.status_dict_vad["chunk_size"] = int(
#                 websocket.status_dict_asr_online["chunk_size"][1] * 60 / websocket.chunk_interval)
#             if len(frames_asr_online) > 0 or len(frames_asr) > 0 or not isinstance(message, str):
#                 if not isinstance(message, str):
#                     frames.append(message)
#                     duration_ms = len(message) // 32
#                     websocket.vad_pre_idx += duration_ms
#
#                     # asr online
#                     frames_asr_online.append(message)
#                     websocket.status_dict_asr_online["is_final"] = speech_end_i != -1
#                     if len(frames_asr_online) % websocket.chunk_interval == 0 or websocket.status_dict_asr_online[
#                         "is_final"]:
#                         if websocket.mode == "2pass" or websocket.mode == "online":
#                             audio_in = b"".join(frames_asr_online)
#                             try:
#                                 await async_asr_online(websocket, audio_in)
#                             except:
#                                 print(f"error in asr streaming, {websocket.status_dict_asr_online}")
#                         frames_asr_online = []
#                     if speech_start:
#                         frames_asr.append(message)
#                     # vad online
#                     try:
#                         speech_start_i, speech_end_i = await async_vad(websocket, message)
#                     except:
#                         print("error in vad")
#                     if speech_start_i != -1:
#                         speech_start = True
#                         beg_bias = (websocket.vad_pre_idx - speech_start_i) // duration_ms
#                         frames_pre = frames[-beg_bias:]
#                         frames_asr = []
#                         frames_asr.extend(frames_pre)
#                 # asr punc offline
#                 if speech_end_i != -1 or not websocket.is_speaking:
#                     # print("vad end point")
#                     if websocket.mode == "2pass" or websocket.mode == "offline":
#                         audio_in = b"".join(frames_asr)
#                         try:
#                             await async_asr(websocket, audio_in)
#                         except:
#                             print("error in asr offline")
#                     frames_asr = []
#                     speech_start = False
#                     frames_asr_online = []
#                     websocket.status_dict_asr_online["cache"] = {}
#                     if not websocket.is_speaking:
#                         websocket.vad_pre_idx = 0
#                         frames = []
#                         websocket.status_dict_vad["cache"] = {}
#                     else:
#                         frames = frames[-20:]
#
#
#     except websockets.ConnectionClosed:
#         print("ConnectionClosed...", websocket_users, flush=True)
#         await ws_reset(websocket)
#         websocket_users.remove(websocket)
#     except websockets.InvalidState:
#         print("InvalidState...")
#     except Exception as e:
#         print("Exception:", e)


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
        rec_result = model_asr.generate(input=audio_in, **websocket.status_dict_asr)[0]
        # print("offline_asr, ", rec_result)
        if model_punc is not None and len(rec_result["text"]) > 0:
            # print("offline, before punc", rec_result, "cache", websocket.status_dict_punc)
            rec_result = model_punc.generate(input=rec_result['text'], **websocket.status_dict_punc)[0]
        # print("offline, after punc", rec_result)
        if len(rec_result["text"]) > 0:
            # print("offline", rec_result)
            mode = "2pass-offline" if "2pass" in websocket.mode else websocket.mode
            message = json.dumps({"mode": mode, "text": rec_result["text"], "wav_name": websocket.wav_name,
                                  "is_final": websocket.is_speaking})
            await websocket.send(message)


async def async_asr_online(websocket, audio_in):
    if len(audio_in) > 0:
        # print(websocket.status_dict_asr_online.get("is_final", False))
        rec_result = model_asr_streaming.generate(input=audio_in, **websocket.status_dict_asr_online)[0]
        # print("online, ", rec_result)
        if websocket.mode == "2pass" and websocket.status_dict_asr_online.get("is_final", False):
            return
        #     websocket.status_dict_asr_online["cache"] = dict()
        if len(rec_result["text"]):
            mode = "2pass-online" if "2pass" in websocket.mode else websocket.mode
            message = json.dumps({"mode": mode, "text": rec_result["text"], "wav_name": websocket.wav_name,
                                  "is_final": websocket.is_speaking})
            await websocket.send(message)

# start_server = websockets.serve(ws_serve, args.host, args.port, subprotocols=["binary"], ping_interval=None)
# asyncio.get_event_loop().run_until_complete(start_server)
# asyncio.get_event_loop().run_forever()


@funasr_router.websocket("/recognition_wss/ali")
async def ws_serve(websocket: WebSocket):
    await websocket.accept()
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
    print("new user connected", flush=True)
    while True:
        message = await websocket.receive()
        message_type = message["type"]
        # 根据接受到的数据类型，对接收信息进行解析
        if message_type == "websocket.disconnect":
            print("ConnectionClosed...", websocket_users, flush=True)
            await ws_reset(websocket)
            websocket_users.remove(websocket)
            break
        elif message_type == "websocket.receive":
            # 字节信息为 音频数据， 文本信息为 初始化信息
            message = message.get("bytes") or message.get("text")

        # 判断收到的是不是第一帧，用于初始化音频处理信息
        if isinstance(message,str):
            messagejson = json.loads(message)
            print('检测通过...开始load json...')
            if "is_speaking" in messagejson:
                websocket.is_speaking = messagejson["is_speaking"]
                websocket.status_dict_asr_online["is_final"] = not websocket.is_speaking
            if "chunk_interval" in messagejson:
                websocket.chunk_interval = messagejson["chunk_interval"]
            if "wav_name" in messagejson:
                websocket.wav_name = messagejson.get("wav_name")
            if "chunk_size" in messagejson:
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
                websocket.mode = messagejson["mode"]
        websocket.status_dict_vad["chunk_size"] = int(
            websocket.status_dict_asr_online["chunk_size"][1] * 60 / websocket.chunk_interval)
        # 如果保存的音频帧超过了0 或者 message 不再是json string格式，说明当前message是音频帧，应该保存
        if len(frames_asr_online) > 0 or len(frames_asr) > 0 or not isinstance(message, str):
            if not isinstance(message, str):
                frames.append(message)
                duration_ms = len(message) // 32
                websocket.vad_pre_idx += duration_ms

                # asr online
                frames_asr_online.append(message)
                websocket.status_dict_asr_online["is_final"] = speech_end_i != -1
                if len(frames_asr_online) % websocket.chunk_interval == 0 or websocket.status_dict_asr_online[
                    "is_final"]:
                    if websocket.mode == "2pass" or websocket.mode == "online":
                        audio_in = b"".join(frames_asr_online)
                        try:
                            await async_asr_online(websocket, audio_in)
                        except:
                            print(f"error in asr streaming, {websocket.status_dict_asr_online}")
                    frames_asr_online = []
                if speech_start:
                    frames_asr.append(message)
                # vad online
                try:
                    speech_start_i, speech_end_i = await async_vad(websocket, message)
                except:
                    print("error in vad")
                if speech_start_i != -1:
                    speech_start = True
                    beg_bias = (websocket.vad_pre_idx - speech_start_i) // duration_ms
                    frames_pre = frames[-beg_bias:]
                    frames_asr = []
                    frames_asr.extend(frames_pre)
            # asr punc offline
            if speech_end_i != -1 or not websocket.is_speaking:
                # print("vad end point")
                if websocket.mode == "2pass" or websocket.mode == "offline":
                    audio_in = b"".join(frames_asr)
                    try:
                        await async_asr(websocket, audio_in)
                    except:
                        print("error in asr offline")
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