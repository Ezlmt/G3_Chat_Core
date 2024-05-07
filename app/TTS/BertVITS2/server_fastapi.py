"""
api服务 多版本多模型 fastapi实现
"""
import logging
import random
from pydantic import BaseModel
import gradio
import numpy as np
from fastapi import Query, Request,APIRouter
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from io import BytesIO
from scipy.io import wavfile
import uvicorn
import torch
import webbrowser
import psutil
import GPUtil
from typing import Dict, List
import os
from urllib.parse import unquote
from app.TTS.BertVITS2.tools.log import logger
from app.TTS.BertVITS2.tools.infer import infer
from app.TTS.BertVITS2.tools import translate as trans
from app.TTS.BertVITS2.re_matching import cut_sent
from app.TTS.BertVITS2 import Model,Models,initial_TTS

from app.TTS.BertVITS2.configs import config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

tts_router = APIRouter()
tts_router.logger = logger
current_dir_path = os.path.dirname(__file__)

# 挂载静态文件
StaticDir: str = os.path.join(current_dir_path,"Web")
dirs = [fir.name for fir in os.scandir(StaticDir) if fir.is_dir()]
files = [fir.name for fir in os.scandir(StaticDir) if fir.is_dir()]
for dirName in dirs:
    print(f"{StaticDir}\\{dirName}")
    tts_router.mount(
        f"/{dirName}",
        StaticFiles(directory=f"{StaticDir}\\{dirName}"),
        name=dirName,
    )
loaded_models = Models()
# 加载模型
models_info = config.server_config.models
for model_info in models_info:
    loaded_models.init_model(
        config_path=os.path.join(current_dir_path,model_info["config"]),
        model_path=os.path.join(current_dir_path,model_info["model"]),
        device=model_info["device"],
        language=model_info["language"],
    )
# 先测试一句，用于初始化前缀词，之后推理速度就会快很多
initial_TTS(sdp_ratio=0.2,
            noise=0.2,
            noisew=0.9,
            length=1,
            speaker_name="ying",
            language="ZH",
            hps=loaded_models.models[1].hps,
            net_g=loaded_models.models[1].net_g,
            device=loaded_models.models[1].device,)
@tts_router.get("/")
async def index():
    html_path = os.path.join(current_dir_path,"Web\\index.html")
    return FileResponse(html_path)

class Text(BaseModel):
    text: str

# 这个是获取音频的Post方法！！！
"""
一次 Post 示例：
http://127.0.0.1:7860/voice?model_id=2&speaker_name=ying&sdp_ratio=0.2&noise=0.2
&noisew=0.9&length=1&language=ZH&auto_translate=false&auto_split=true
"""
@tts_router.post("/voice")
def voice(
    request: Request,  # fastapi自动注入
    text: Text,
    model_id: int = Query(..., description="模型ID"),  # 模型序号
    speaker_name: str = Query(
        None, description="说话人名"
    ),  # speaker_name与 speaker_id二者选其一
    speaker_id: int = Query(None, description="说话人id，与speaker_name二选一"),
    sdp_ratio: float = Query(0.2, description="SDP/DP混合比"),
    noise: float = Query(0.2, description="感情"),
    noisew: float = Query(0.9, description="音素长度"),
    length: float = Query(1, description="语速"),
    language: str = Query(None, description="语言"),  # 若不指定使用语言则使用默认值
    auto_translate: bool = Query(False, description="自动翻译"),
    auto_split: bool = Query(False, description="自动切分"),
):
    print("接收到get请求，正在合成语音...")
    """语音接口"""
    text = text.text
    logger.info(
        f"{request.client.host}:{request.client.port}/voice  { unquote(str(request.query_params) )} text={text}"
    )
    # 检查模型是否存在
    if model_id not in loaded_models.models.keys():
        return {"status": 10, "detail": f"模型model_id={model_id}未加载"}
    # 检查是否提供speaker
    if speaker_name is None and speaker_id is None:
        return {"status": 11, "detail": "请提供speaker_name或speaker_id"}
    elif speaker_name is None:
        # 检查speaker_id是否存在
        if speaker_id not in loaded_models.models[model_id].id2spk.keys():
            return {"status": 12, "detail": f"角色speaker_id={speaker_id}不存在"}
        speaker_name = loaded_models.models[model_id].id2spk[speaker_id]
    # 检查speaker_name是否存在
    if speaker_name not in loaded_models.models[model_id].spk2id.keys():
        return {"status": 13, "detail": f"角色speaker_name={speaker_name}不存在"}
    if language is None:
        language = loaded_models.models[model_id].language
    if auto_translate:
        text = trans.translate(Sentence=text, to_Language=language.lower())
    if not auto_split:
        with torch.no_grad():
            audio = infer(
                text=text,
                sdp_ratio=sdp_ratio,
                noise_scale=noise,
                noise_scale_w=noisew,
                length_scale=length,
                sid=speaker_name,
                language=language,
                hps=loaded_models.models[model_id].hps,
                net_g=loaded_models.models[model_id].net_g,
                device=loaded_models.models[model_id].device,
            )
    else:
        texts = cut_sent(text)
        audios = []
        with torch.no_grad():
            for t in texts:
                audios.append(
                    infer(
                        text=t,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise,
                        noise_scale_w=noisew,
                        length_scale=length,
                        sid=speaker_name,
                        language=language,
                        hps=loaded_models.models[model_id].hps,
                        net_g=loaded_models.models[model_id].net_g,
                        device=loaded_models.models[model_id].device,
                    )
                )
            audios.append(np.zeros((int)(44100 * 0.3)))
            audio = np.concatenate(audios)
            audio = gradio.processing_utils.convert_to_16_bit_wav(audio)
    wavContent = BytesIO()
    wavfile.write(
        wavContent, loaded_models.models[model_id].hps.data.sampling_rate, audio
    )
    response = Response(content=wavContent.getvalue(), media_type="audio/wav")
    return response

@tts_router.get("/voice")
def voice(
    request: Request,  # fastapi自动注入
    text: str = Query(..., description="输入文字"),
    model_id: int = Query(..., description="模型ID"),  # 模型序号
    speaker_name: str = Query(
        None, description="说话人名"
    ),  # speaker_name与 speaker_id二者选其一
    speaker_id: int = Query(None, description="说话人id，与speaker_name二选一"),
    sdp_ratio: float = Query(0.2, description="SDP/DP混合比"),
    noise: float = Query(0.2, description="感情"),
    noisew: float = Query(0.9, description="音素长度"),
    length: float = Query(1, description="语速"),
    language: str = Query(None, description="语言"),  # 若不指定使用语言则使用默认值
    auto_translate: bool = Query(False, description="自动翻译"),
    auto_split: bool = Query(False, description="自动切分"),
):
    """语音接口"""
    logger.info(
        f"{request.client.host}:{request.client.port}/voice  { unquote(str(request.query_params) )}"
    )
    # 检查模型是否存在
    if model_id not in loaded_models.models.keys():
        return {"status": 10, "detail": f"模型model_id={model_id}未加载"}
    # 检查是否提供speaker
    if speaker_name is None and speaker_id is None:
        return {"status": 11, "detail": "请提供speaker_name或speaker_id"}
    elif speaker_name is None:
        # 检查speaker_id是否存在
        if speaker_id not in loaded_models.models[model_id].id2spk.keys():
            return {"status": 12, "detail": f"角色speaker_id={speaker_id}不存在"}
        speaker_name = loaded_models.models[model_id].id2spk[speaker_id]
    # 检查speaker_name是否存在
    if speaker_name not in loaded_models.models[model_id].spk2id.keys():
        return {"status": 13, "detail": f"角色speaker_name={speaker_name}不存在"}
    if language is None:
        language = loaded_models.models[model_id].language
    if auto_translate:
        text = trans.translate(Sentence=text, to_Language=language.lower())
    if not auto_split:
        with torch.no_grad():
            audio = infer(
                text=text,
                sdp_ratio=sdp_ratio,
                noise_scale=noise,
                noise_scale_w=noisew,
                length_scale=length,
                sid=speaker_name,
                language=language,
                hps=loaded_models.models[model_id].hps,
                net_g=loaded_models.models[model_id].net_g,
                device=loaded_models.models[model_id].device,
            )
    else:
        texts = cut_sent(text)
        audios = []
        with torch.no_grad():
            for t in texts:
                audios.append(
                    infer(
                        text=t,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise,
                        noise_scale_w=noisew,
                        length_scale=length,
                        sid=speaker_name,
                        language=language,
                        hps=loaded_models.models[model_id].hps,
                        net_g=loaded_models.models[model_id].net_g,
                        device=loaded_models.models[model_id].device,
                    )
                )
            audios.append(np.zeros((int)(44100 * 0.3)))
            audio = np.concatenate(audios)
            audio = gradio.processing_utils.convert_to_16_bit_wav(audio)
    wavContent = BytesIO()
    wavfile.write(
        wavContent, loaded_models.models[model_id].hps.data.sampling_rate, audio
    )
    response = Response(content=wavContent.getvalue(), media_type="audio/wav")
    return response

@tts_router.get("/models/info")
def get_loaded_models_info(request: Request):
    """获取已加载模型信息"""

    result: Dict[str, Dict] = dict()
    for key, model in loaded_models.models.items():
        result[str(key)] = model.to_dict()
    return result

@tts_router.get("/models/delete")
def delete_model(
    request: Request, model_id: int = Query(..., description="删除模型id")
):
    """删除指定模型"""
    logger.info(
        f"{request.client.host}:{request.client.port}/models/delete  { unquote(str(request.query_params) )}"
    )
    result = loaded_models.del_model(model_id)
    if result is None:
        return {"status": 14, "detail": f"模型{model_id}不存在，删除失败"}
    return {"status": 0, "detail": "删除成功"}

    """
        tts_router: /html/main.
        根据页面返回的model_path和language等信息，添加模型
        Args:
            model_path
            config_path
            language
            device
        Returns:
            json
    """
@tts_router.get("/models/add")
def add_model(
    request: Request,
    model_path: str = Query(..., description="添加模型路径"),
    config_path: str = Query(
        None, description="添加模型配置文件路径，不填则使用./config.json或../config.json"
    ),
    device: str = Query("cuda", description="推理使用设备"),
    language: str = Query("ZH", description="模型默认语言"),
):
    """添加指定模型：允许重复添加相同路径模型，且不重复占用内存"""
    logger.info(
        f"{request.client.host}:{request.client.port}/models/add  { unquote(str(request.query_params) )}"
    )
    if os.path.isfile(model_path) is False:
        model_path = os.path.join(current_dir_path,model_path.replace("/", "\\"))
        print(f"[当前的model_path为]: {model_path}，[当前的文件夹路径为：]{current_dir_path}")
    if config_path is None:
        model_dir = os.path.dirname(model_path)
        model_config_path1 = os.path.join(model_dir,"config,json")
        model_config_path2 = os.path.join(os.path.dirname(model_dir),"config.json")
        if os.path.isfile(model_config_path1):
            config_path = model_config_path1
        elif os.path.isfile(model_config_path2):
            config_path = model_config_path2
        else:
            print(f"模型添加失败!配置文件路径不存在！路径信息：{model_config_path1}")
            return {
                "status": 15,
                "detail": "查询未传入配置文件路径，同时默认路径./与../中不存在配置文件config.json。",
            }
    try:
        model_id = loaded_models.init_model(
            config_path=config_path,
            model_path=model_path,
            device=device,
            language=language,
        )
    except Exception:
        logging.exception("模型加载出错")
        return {
            "status": 16,
            "detail": "模型加载出错，详细查看日志",
        }
    return {
        "status": 0,
        "detail": "模型添加成功",
        "Data": {
            "model_id": model_id,
            "model_info": loaded_models.models[model_id].to_dict(),
        },
    }

def _get_all_models(root_dir: str = "Data", only_unloaded: bool = False):
    """从root_dir搜索获取所有可用模型"""
    result: Dict[str, List[str]] = dict()
    root_dir=os.path.join(current_dir_path,root_dir)
    files = os.listdir(root_dir) + ["."]
    for file in files:
        if os.path.isdir(os.path.join(root_dir, file)):
            sub_dir = os.path.join(root_dir, file)
            # 搜索 "sub_dir" 、 "sub_dir/models" 两个路径
            result[file] = list()
            sub_files = os.listdir(sub_dir)
            model_files = []
            for sub_file in sub_files:
                relpath = os.path.realpath(os.path.join(sub_dir, sub_file))
                if only_unloaded and relpath in loaded_models.path2ids.keys():
                    continue
                if sub_file.endswith(".pth") and sub_file.startswith("G_"):
                    if os.path.isfile(relpath):
                        model_files.append(sub_file)
            # 对模型文件按步数排序
            model_files = sorted(
                model_files,
                key=lambda pth: int(pth.lstrip("G_").rstrip(".pth"))
                if pth.lstrip("G_").rstrip(".pth").isdigit()
                else 10**10,
            )
            result[file] = model_files
            models_dir = os.path.join(sub_dir, "models")
            model_files = []
            if os.path.isdir(models_dir):
                sub_files = os.listdir(models_dir)
                for sub_file in sub_files:
                    relpath = os.path.realpath(os.path.join(models_dir, sub_file))
                    if only_unloaded and relpath in loaded_models.path2ids.keys():
                        continue
                    if sub_file.endswith(".pth") and sub_file.startswith("G_"):
                        if os.path.isfile(os.path.join(models_dir, sub_file)):
                            model_files.append(f"models/{sub_file}")
                # 对模型文件按步数排序
                model_files = sorted(
                    model_files,
                    key=lambda pth: int(pth.lstrip("models/G_").rstrip(".pth"))
                    if pth.lstrip("models/G_").rstrip(".pth").isdigit()
                    else 10**10,
                )
                result[file] += model_files
            if len(result[file]) == 0:
                result.pop(file)

    return result

@tts_router.get("/models/get_unloaded")
def get_unloaded_models_info(
    request: Request, root_dir: str = Query("Data", description="搜索根目录")
):
    """获取未加载模型"""
    logger.info(
        f"{request.client.host}:{request.client.port}/models/get_unloaded  { unquote(str(request.query_params) )}"
    )
    return _get_all_models(root_dir, only_unloaded=True)

@tts_router.get("/models/get_local")
def get_local_models_info(
    request: Request, root_dir: str = Query("Data", description="搜索根目录")
):
    """获取全部本地模型"""
    logger.info(
        f"{request.client.host}:{request.client.port}/models/get_local  { unquote(str(request.query_params) )}"
    )
    return _get_all_models(root_dir, only_unloaded=False)

@tts_router.get("/status")
def get_status():
    """获取电脑运行状态"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_total = memory_info.total
    memory_available = memory_info.available
    memory_used = memory_info.used
    memory_percent = memory_info.percent
    gpuInfo = []
    devices = ["cpu"]
    for i in range(torch.cuda.device_count()):
        devices.append(f"cuda:{i}")
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        gpuInfo.append(
            {
                "gpu_id": gpu.id,
                "gpu_load": gpu.load,
                "gpu_memory": {
                    "total": gpu.memoryTotal,
                    "used": gpu.memoryUsed,
                    "free": gpu.memoryFree,
                },
            }
        )
    return {
        "devices": devices,
        "cpu_percent": cpu_percent,
        "memory_total": memory_total,
        "memory_available": memory_available,
        "memory_used": memory_used,
        "memory_percent": memory_percent,
        "gpu": gpuInfo,
    }

@tts_router.get("/tools/translate")
def translate(
    request: Request,
    texts: str = Query(..., description="待翻译文本"),
    to_language: str = Query(..., description="翻译目标语言"),
):
    """翻译"""
    logger.info(
        f"{request.client.host}:{request.client.port}/tools/translate  { unquote(str(request.query_params) )}"
    )
    return {"texts": trans.translate(Sentence=texts, to_Language=to_language)}

all_examples: Dict[str, Dict[str, List]] = dict()  # 存放示例

@tts_router.get("/tools/random_example")
def random_example(
    request: Request,
    language: str = Query(None, description="指定语言，未指定则随机返回"),
    root_dir: str = Query("Data", description="搜索根目录"),
):
    """
    获取一个随机音频+文本，用于对比，音频会从本地目录随机选择。
    """
    logger.info(
        f"{request.client.host}:{request.client.port}/tools/random_example  { unquote(str(request.query_params) )}"
    )
    global all_examples
    # 数据初始化
    if root_dir not in all_examples.keys():
        all_examples[root_dir] = {"ZH": [], "JP": [], "EN": []}

        examples = all_examples[root_dir]

        # 从项目Data目录中搜索train/val.list
        for root, directories, _files in os.walk("Data"):
            for file in _files:
                if file in ["train.list", "val.list"]:
                    with open(
                        os.path.join(root, file), mode="r", encoding="utf-8"
                    ) as f:
                        lines = f.readlines()
                        for line in lines:
                            data = line.split("|")
                            if len(data) != 7:
                                continue
                            # 音频存在 且语言为ZH/EN/JP
                            if os.path.isfile(data[0]) and data[2] in [
                                "ZH",
                                "JP",
                                "EN",
                            ]:
                                examples[data[2]].append(
                                    {
                                        "text": data[3],
                                        "audio": data[0],
                                        "speaker": data[1],
                                    }
                                )

    examples = all_examples[root_dir]
    if language is None:
        if len(examples["ZH"]) + len(examples["JP"]) + len(examples["EN"]) == 0:
            return {"status": 17, "detail": "没有加载任何示例数据"}
        else:
            # 随机选一个
            rand_num = random.randint(
                0,
                len(examples["ZH"]) + len(examples["JP"]) + len(examples["EN"]) - 1,
            )
            # ZH
            if rand_num < len(examples["ZH"]):
                return {"status": 0, "Data": examples["ZH"][rand_num]}
            # JP
            if rand_num < len(examples["ZH"]) + len(examples["JP"]):
                return {
                    "status": 0,
                    "Data": examples["JP"][rand_num - len(examples["ZH"])],
                }
            # EN
            return {
                "status": 0,
                "Data": examples["EN"][
                    rand_num - len(examples["ZH"]) - len(examples["JP"])
                ],
            }

    else:
        if len(examples[language]) == 0:
            return {"status": 17, "detail": f"没有加载任何{language}数据"}
        return {
            "status": 0,
            "Data": examples[language][
                random.randint(0, len(examples[language]) - 1)
            ],
        }

@tts_router.get("/tools/get_audio")
def get_audio(request: Request, path: str = Query(..., description="本地音频路径")):
    logger.info(
        f"{request.client.host}:{request.client.port}/tools/get_audio  { unquote(str(request.query_params) )}"
    )
    if not os.path.isfile(path):
        return {"status": 18, "detail": "指定音频不存在"}
    if not path.endswith(".wav"):
        return {"status": 19, "detail": "非wav格式文件"}
    return FileResponse(path=path)

if __name__ == "__main__":
    webbrowser.open(f"http://127.0.0.1:{config.server_config.port}")
    uvicorn.run(
        tts_router, port=config.server_config.port, host="0.0.0.0", log_level="warning"
    )
