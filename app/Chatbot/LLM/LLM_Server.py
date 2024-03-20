from fastapi import APIRouter, Request, Form, Response, WebSocket, WebSocketDisconnect
from fastapi import FastAPI, Depends, HTTPException
from app.configs.base_configs import get_config_instance
from fastapi.responses import JSONResponse
from app.Chatbot.LLM.Spark import get_spark_instance,logger
from fastapi.responses import FileResponse
import json
from app.Database.Character_Manager import get_character_instance
import websocket
from starlette.templating import Jinja2Templates
from websockets.exceptions import ConnectionClosedOK
import re
import requests
import asyncio
from datetime import datetime
from app.utils.utils import contains_only_punctuation, checkFullSentence, split_sentences

llm_router = APIRouter()
@llm_router.post("/updateChar")
async def update_char(request: Request, char_instance = Depends(get_character_instance)):
    """
    Chatbot API /updateChar.
    接收角色英文名request，”name“关键字. 然后server更新char_instance的角色数据。

    Args:
        request (Request): request需要”name“关键字.

    """
    data = await request.json()
    CharName = data['name']
    char_instance.updateChar(CharName)

@llm_router.websocket("/sparkws")
async def websocket_endpoint(
        websocket: WebSocket,
        char_instance = Depends(get_character_instance),
        SparkChat_instance = Depends(get_spark_instance),
        config = Depends(get_config_instance)
    ):
    """
    Chatbot websocket /ws.

    Args:
        websocket: 等待 websockets.connect(ws_url).

    Returns:
        str:
            sentence：经过标点分句后完整的句子
    """
    await websocket.accept()

    while True:
        logger.info(f"websocket.accept")
        answer = ""
        # 接收用户查询
        message = await websocket.receive()
        message_type = message["type"]
        # 根据接受到的数据类型，对接收信息进行解析
        if message_type == "websocket.receive":
            receive_msg = message.get("bytes") or message.get("text")
            if isinstance(receive_msg,bytes):
                data = receive_msg.decode('utf-8')
            elif isinstance(receive_msg,str):
                data = receive_msg
        elif message_type == "websocket.disconnect":
            logger.info(f"websocket.disconnect")
            break
        # data = await websocket.receive_text()
        logger.info(f"收到提问: {data}")
        # 在这里与LLM交互，获取iterator chunks，存储流式回复
        try:
            chunks = SparkChat_instance.run_chat_stream(data)
        except Exception as e:
            logger.info(f"请求异常: {e}")
            chunks = '你说的我不太了解啦，你可以换一种方式说吗？'
        for chunk in chunks:
            answer += chunk
            await websocket.send_text(chunk)
            await asyncio.sleep(0)
        logger.info(f"角色回复：{answer}")
        # 发送结束，发送quit标志
        await websocket.send_text("quit")
