from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.ASR.Whisper.ASR_Server import asr_router,static_file_path,templates_file_path
from app.configs import base_config
import uvicorn
import webbrowser
asr_app = FastAPI()
asr_app.mount('/static', StaticFiles(directory=static_file_path), name='static')
asr_app.include_router(asr_router)

if __name__ == '__main__':
    print("本地ASR服务，请勿将服务端口暴露于外网")
    print(f"WebAPI地址: http://127.0.0.1:{base_config.asr_port}")
    print("\n-------------------------ASR服务启动完毕-----------------------------\n")
    uvicorn.run(
        asr_app, port=int(base_config.asr_port), host="0.0.0.0", log_level="warning"
    )