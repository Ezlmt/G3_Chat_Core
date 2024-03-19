import os
import argparse
from tqdm import tqdm
import sys
import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.hub.snapshot_download import snapshot_download
from common.constants import Languages
from common.log import logger
from common.stdout_wrapper import SAFE_STDOUT
import soundfile

# 指定本地目录地址
local_dir_root = "./models"
# model_dir = snapshot_download('iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch', cache_dir=local_dir_root)
# model_dir_vad = snapshot_download('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', cache_dir=local_dir_root)
# model_dir_punc_ct = snapshot_download('iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch', cache_dir=local_dir_root)
#
# inference_pipeline = pipeline(
#     task=Tasks.auto_speech_recognition,
#     model=model_dir,model_revision="v2.0.4",
#     vad_model=model_dir_vad,vad_model_revision="v2.0.4",
#     punc_model=model_dir_punc_ct,punc_model_revision="v2.0.4",
#     #lm_model='damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch',
#     #lm_weight=0.15,
#     #beam_size=10,
# )
#
#
# param_dict = {}
# param_dict['use_timestamp'] = False
# # folderpath = sys.argv[1]
# extensions = ['wav']
#
# speech,sample_rate = soundfile.read('./audiofile/test.wav')
#
# def transcribe_audio(audio,language):
#     if language == 'zh':
#         rec_result = inference_pipeline(audio_in=audio, param_dict=param_dict)
#     print(f"识别结果: {rec_result}")



# _________________________________________test Whisper____________________________________________
from faster_whisper import WhisperModel
from io import BytesIO
import typing
import io
import collections
import wave
import datetime
from fastapi import FastAPI,WebSocket,WebSocketDisconnect,\
    WebSocketException,Request,UploadFile, File
import uvicorn
from websockets.exceptions import ConnectionClosedOK

import pyaudio
import webrtcvad
import logging

bart_path = snapshot_download('damo/nlp_bart_text-error-correction_chinese', cache_dir=local_dir_root)
punc_ct_path = snapshot_download('iic/punc_ct-transformer_cn-en-common-vocab471067-large', cache_dir=local_dir_root)
print(punc_ct_path)

# 建立 文本纠错、标点预测 pipeline
word_fix_pipeline = pipeline(Tasks.text_error_correction, model=bart_path, model_revision='v1.0.1')
punc_ct_pipeline = pipeline(Tasks.punctuation,model=punc_ct_path,model_revision="v2.0.4")


class Transcriber(object):
    def __init__(self,
                 model_size: str,
                 device: str = "auto",
                 compute_type: str = "default",
                 prompt: str = '实时/低延迟语音转写服务，林黛玉、倒拔、杨柳树、鲁迅、周树人、关键词、转写正确'
                 ) -> None:
        """ FasterWhisper 语音转写

        Args:
            model_size (str): 模型大小，可选项为 "tiny", "base", "small", "medium", "large" 。
                更多信息参考：https://github.com/openai/whisper
            device (str, optional): 模型运行设备。
            compute_type (str, optional): 计算类型。默认为"default"。
            prompt (str, optional): 初始提示。如果需要转写简体中文，可以使用简体中文提示。
        """

        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.prompt = prompt

    def __enter__(self) -> 'Transcriber':
        self._model = WhisperModel(self.model_size,
                                  device=self.device,
                                  compute_type=self.compute_type)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass

    def __call__(self, audio: bytes) -> typing.Generator[str, None, None]:
        segments, info = self._model.transcribe(BytesIO(audio),
                                               initial_prompt=self.prompt)
        if info.language != "zh":
            return {"error": "transcribe Chinese only"}
        for segment in segments:
            t = segment.text
            if t.strip().replace('.', ''):
                yield t

def get_time():
    return datetime.datetime.now().strftime("%H:%M:%S")

if __name__ == '__main__':
    with open('./audiofile/test.wav', 'rb') as f:
        audio_bytes = f.read()
        with Transcriber(model_size="large") as transcriber:
            transcriptions = transcriber(audio_bytes)  # 调用transcriber对象进行转写，这样才是调用__call__方法
            sentence = ''
            for transcription in transcriptions:
                sentence += transcription
            print(f"识别结果: {sentence}")
            predict = punc_ct_pipeline(sentence)
            print(f"预测结果: {predict[0]['text']},结果格式：{type(predict)}")
            refix = word_fix_pipeline(predict[0]['text'])
            print(f"修正结果: {refix}")

    # print(word_fix_pipeline("今天天气好！喜阳阳楼！"))