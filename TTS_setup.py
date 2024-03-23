from fastapi import FastAPI
from app.TTS.BertVITS2.server_fastapi import tts_router,logger
# from app.ASR.Whisper.ASR_Server import asr_router
from app.configs import base_config
import uvicorn
import webbrowser

tts_app = FastAPI()
tts_app.include_router(tts_router)

if __name__ == '__main__':
    logger.warning("本地TTS服务，请勿将服务端口暴露于外网")
    logger.info(f"WebAPI地址: http://127.0.0.1:{base_config.tts_port}")
    print("\n-------------------------TTS服务启动完毕-----------------------------\n")
    uvicorn.run(
        tts_app, port=int(base_config.tts_port), host="0.0.0.0", log_level="warning"
    )