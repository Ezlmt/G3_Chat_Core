import argparse
import asyncio
import functools
import json
import os
from io import BytesIO

import uvicorn
from fastapi import FastAPI, BackgroundTasks, File, Body, UploadFile, Request,APIRouter
from fastapi.responses import StreamingResponse
from fastapi.logger import logger
from faster_whisper import WhisperModel
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from zhconv import convert
from app.configs.base_configs import get_config_instance
from .utils.data_utils import remove_punctuation
from .utils.utils import add_arguments, print_arguments

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
app_config = get_config_instance()
# 参数 定义
model_path = app_config.asr_model_path
use_gpu = bool(app_config.use_gpu)
use_int8 = bool(app_config.use_int8)
local_files_only = bool(app_config.local_files_only)
num_workers = int(app_config.num_workers)
vad_filter = bool(app_config.vad_filter)
current_dir_path = os.path.dirname(__file__)
static_file_path = os.path.join(current_dir_path,"static")
templates_file_path = os.path.join(current_dir_path,"templates")

# router 定义
asr_router = APIRouter()

templates = Jinja2Templates(directory=templates_file_path)
model_semaphore = None



# 检查模型文件是否存在
assert os.path.exists(model_path), f"模型文件{model_path}不存在"
# 加载模型
if use_gpu:
    if not use_int8:
        model = WhisperModel(model_path, device="cuda", compute_type="float16", num_workers=num_workers,
                             local_files_only=local_files_only)
    else:
        model = WhisperModel(model_path, device="cuda", compute_type="int8_float16", num_workers=num_workers,
                             local_files_only=local_files_only)
else:
    model = WhisperModel(model_path, device="cpu", compute_type="int8", num_workers=num_workers,
                         local_files_only=local_files_only)
# 预热
_, _ = model.transcribe(current_dir_path+"\\dataset\\test.wav", beam_size=5)

def release_model_semaphore():
    model_semaphore.release()

# 进行语音识别的函数
def recognition(file: File, to_simple: int, remove_pun: int, language: str = "zh", task: str = "transcribe"):
    # beam_size被设置为10，即束搜索算法会保留10个候选序列进行评分和选择。
    segments, info = model.transcribe(file, beam_size=10, task=task, vad_filter=vad_filter)
    for segment in segments:
        text = segment.text
        if to_simple == 1:
            text = convert(text, 'zh-cn')
        # 是否要去除标点符号
        if remove_pun == 1:
            text = remove_punctuation(text)
        ret = {"result": text, "start": round(segment.start, 2), "end": round(segment.end, 2)}
        # 返回json格式的字节
        yield json.dumps(ret).encode() + b"\0"

# 流式语音识别
@asr_router.post("/recognition_stream")
async def api_recognition_stream(to_simple: int = Body(1, description="是否繁体转简体", embed=True),
                                 remove_pun: int = Body(0, description="是否删除标点符号", embed=True),
                                 language: str = Body("zh", description="设置语言，简写，如果不指定则自动检测语言", embed=True),
                                 task: str = Body("transcribe", description="识别任务类型，支持transcribe和translate", embed=True),
                                 audio: UploadFile = File(..., description="音频文件")):
    global model_semaphore
    if language == "None": language = None
    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(5)
    await model_semaphore.acquire()
    contents = await audio.read()
    data = BytesIO(contents)
    generator = recognition(file=data, to_simple=to_simple, remove_pun=remove_pun, language=language, task=task)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    # 返回一个iterator，进行流式返回。
    return StreamingResponse(generator, background=background_tasks)

# 语音识别
@asr_router.post("/recognition")
async def api_recognition(to_simple: int = Body(1, description="是否繁体转简体", embed=True),
                          remove_pun: int = Body(0, description="是否删除标点符号", embed=True),
                          language: str = Body("zh", description="设置语言，简写，如果不指定则自动检测语言", embed=True),
                          task: str = Body("transcribe", description="识别任务类型，支持transcribe和translate", embed=True),
                          audio: UploadFile = File(..., description="音频文件")):
    if language == "None":language=None
    contents = await audio.read()
    data = BytesIO(contents)
    generator = recognition(file=data, to_simple=to_simple, remove_pun=remove_pun, language=language, task=task)
    results = []
    sentence = ""
    for output in generator:
        output = json.loads(output[:-1].decode("utf-8"))
        results.append(output)
        sentence += output["result"]
    ret = {"text":sentence,"segments": results,"language":language}
    return ret


@asr_router.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "id": id})


if __name__ == '__main__':
    uvicorn.run(asr_router, host="0.0.0.0", port=int(app_config.asr_port))
