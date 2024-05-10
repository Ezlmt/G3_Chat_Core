# 一、先下载bert模型
from modelscope import snapshot_download
from app.configs import base_config
import os
# 先确定有Bert模型
local_path = os.path.dirname(__file__)
snapshot_download(base_config.TTSBertModelPath,cache_dir=os.path.join(local_path,base_config.TTSModelDir))

# 二、开始加载Router并启动服务
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.TTS.BertVITS2v202.server_fastapi import TTSRouter,logger
import uvicorn
import webbrowser

tts_app = FastAPI()
tts_app.include_router(TTSRouter)
# 挂载静态文件
StaticDir: str = "./Web"
dirs = [fir.name for fir in os.scandir(StaticDir) if fir.is_dir()]
files = [fir.name for fir in os.scandir(StaticDir) if fir.is_dir()]
for dirName in dirs:
    tts_app.mount(
        f"/{dirName}",
        StaticFiles(directory=f"./{StaticDir}/{dirName}"),
        name=dirName,
    )

if __name__ == '__main__':
    port = int(base_config.tts_port)
    host = "127.0.0.1"
    log_level = "warning"
    logger.warning("本地TTS服务，请勿将服务端口暴露于外网")
    logger.info(f"WebAPI地址: http://{host}:{port}")
    print("\n-------------------------TTS服务启动完毕-----------------------------\n")
    uvicorn.run(
        tts_app, port=port, host=host, log_level=log_level
    )