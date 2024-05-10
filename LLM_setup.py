# 一、先下载bert模型
from modelscope import snapshot_download
from app.configs import base_config
import os
# 先确定有Bert模型
local_path = os.path.dirname(__file__)
print(os.path.join(local_path,base_config.Model_DIR))
snapshot_download(base_config.local_embed_model.replace("\\","/"),cache_dir=os.path.join(local_path,base_config.Model_DIR))


# 二、开始加载Router并启动服务
from fastapi import FastAPI
from app.Chatbot.LLM.LLM_Server import llm_router
from app.Chatbot.LLM.Spark import SparkChatService
from app.Database.Character_Manager import CharacterManager
from app.configs import base_config,AppConfig
from app.utils.logger import get_logger
from app.utils.utils import GetIPaddress
import uvicorn
import webbrowser

logger = get_logger(__name__)
# 获取 config 的实例
AppConfig.initialize()
config = AppConfig.get_instance()

# 实例化CharacterManager类
CharacterManager.initialize(config,config.CharName)
char_instance = CharacterManager.get_instance()

#为SparkChatService初始化
SparkChatService.initialize(config,char_instance)


llm_app = FastAPI()
llm_app.include_router(llm_router)

if __name__ == '__main__':
    logger.warning("SparkLLM服务，请勿将服务端口暴露于外网")
    logger.info(f"WebAPI地址: ws://127.0.0.1:{7980}")
    print("\n-------------------------LLM服务启动完毕-----------------------------\n")
    uvicorn.run(
        llm_app, port=int(7980), host="0.0.0.0", log_level="warning"
    )