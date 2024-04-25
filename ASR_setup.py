from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
# from app.ASR.Whisper.ASR_Server import asr_router,static_file_path,templates_file_path
from app.ASR import funasr_router
from app.configs import base_config
from app.utils.utils import GetIPaddress
import uvicorn
import webbrowser
asr_app = FastAPI()
asr_app.include_router(funasr_router)
# asr_app.mount('/static', StaticFiles(directory=static_file_path), name='static')
# asr_app.include_router(asr_router)

if __name__ == '__main__':
    port = int(base_config.asr_port)
    host = GetIPaddress()
    log_level = "warning"
    print("本地ASR服务，请勿将服务端口暴露于外网")
    print(f"WebAPI地址: ws://{host}:{port}")
    print(f"路由名称：recognition_wss/ali")
    print("\n-------------------------ASR服务启动完毕-----------------------------\n")
    uvicorn.run(
        asr_app, port=port, host=host, log_level=log_level
    )