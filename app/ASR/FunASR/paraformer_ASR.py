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
import datetime

# https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary

# 指定本地目录地址
local_dir_root = "./models"
model_dir = snapshot_download('iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch', cache_dir=local_dir_root)
model_dir_vad = snapshot_download('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', cache_dir=local_dir_root)
model_dir_punc_ct = snapshot_download('iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch', cache_dir=local_dir_root)

# model_dir = local_dir_root+'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
# model_dir_vad = local_dir_root+'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch'
# model_dir_punc_ct = local_dir_root+'iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch'

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model=model_dir,model_revision="v2.0.4",
    vad_model=model_dir_vad,vad_model_revision="v2.0.4",
    punc_model=model_dir_punc_ct,punc_model_revision="v2.0.4",
    #lm_model='damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch',
    #lm_weight=0.15,
    #beam_size=10,
)


param_dict = {}
param_dict['use_timestamp'] = False
# folderpath = sys.argv[1]
extensions = ['wav']

def transcribe_audio(audio,language):
    if language == 'zh':
        rec_result = inference_pipeline(audio_in=audio)
    print(f"识别结果: {rec_result}")

def get_time():
    return datetime.datetime.now().strftime("%H:%M:%S")

if __name__ == '__main__':
    # print(f"当前时间： {get_time()}")
    # with open("./audiofile/test.wav", 'rb') as file:
    #     speech = file
    #     transcribe_audio("speech", "zh")

    transcribe_audio("./audiofile/test.wav","zh")
    print(f"当前时间： {get_time()}")