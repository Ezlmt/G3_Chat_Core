import argparse
import functools
import os
import time
from faster_whisper import WhisperModel

from utils.utils import print_arguments, add_arguments

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("audio_path",  type=str,  default="dataset/test.wav",        help="预测的音频路径")
add_arg("model_path",  type=str,  default="models/whisper-tiny-finetune-ct2", help="转换后的模型路径，转换方式看文档")
add_arg("language",    type=str,  default="zh",   help="设置语言，必须简写，如果为None则自动检测语言")
add_arg("use_gpu",     type=bool, default=True,   help="是否使用gpu进行预测")
add_arg("use_int8",    type=bool, default=False,  help="是否使用int8进行预测")
add_arg("beam_size",   type=int,  default=10,     help="解码搜索大小")
add_arg("num_workers", type=int,  default=1,      help="预测器的并发数量")
add_arg("vad_filter",  type=bool, default=False,  help="是否使用VAD过滤掉部分没有讲话的音频")
add_arg("local_files_only", type=bool, default=True, help="是否只在本地加载模型，不尝试下载")
args = parser.parse_args()
print_arguments(args)

# 检查模型文件是否存在
assert os.path.exists(args.model_path), f"模型文件{args.model_path}不存在"
# 加载模型
if args.use_gpu:
    if not args.use_int8:
        model = WhisperModel(args.model_path, device="cuda", compute_type="float16", num_workers=args.num_workers,
                             local_files_only=args.local_files_only)
    else:
        model = WhisperModel(args.model_path, device="cuda", compute_type="int8_float16", num_workers=args.num_workers,
                             local_files_only=args.local_files_only)
else:
    model = WhisperModel(args.model_path, device="cpu", compute_type="int8", num_workers=args.num_workers,
                         local_files_only=args.local_files_only)
# 预热
_, _ = model.transcribe("dataset/test.wav", beam_size=5)


strat_time = time.time()
with open(args.audio_path,"rb") as audio:
    from io import BytesIO
    audio_bytes = audio.read()
    audio_bytes = bytes(audio_bytes)
    audio_bytes =BytesIO(audio_bytes)
    # 语音识别
    segments, info = model.transcribe(audio_bytes, beam_size=args.beam_size,language=args.language)
sentence = ""
for segment in segments:
    sentence += segment.text
    text = segment.text
    # round(x,y) 对 x 以保留y位小数 进行四舍五入
    print(f"[{round(segment.start, 2)} - {round(segment.end, 2)}]：{text}\n")
end_time = time.time()
print(f"推理耗时: {end_time-strat_time} 秒,回复内容为：{sentence}")
