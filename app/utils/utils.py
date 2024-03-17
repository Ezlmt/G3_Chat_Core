import asyncio
from time import perf_counter
from typing import List, Optional, Callable
from starlette.websockets import WebSocket, WebSocketState
import re

class Singleton:
    _instances = {}

    @classmethod
    def get_instance(cls, *args, **kwargs):
        """ Static access method. """
        if cls not in cls._instances:
            cls._instances[cls] = cls(*args, **kwargs)

        return cls._instances[cls]

    @classmethod
    def initialize(cls, *args, **kwargs):
        """ Static access method. """
        if cls not in cls._instances:
            cls._instances[cls] = cls(*args, **kwargs)


class ConnectionManager(Singleton):
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"Client #{id(websocket)} left the chat")
        # await self.broadcast_message(f"Client #{id(websocket)} left the chat")

    async def send_message(self, message: str, websocket: WebSocket):
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_text(message)

    async def broadcast_message(self, message: str):
        for connection in self.active_connections:
            if connection.application_state == WebSocketState.CONNECTED:
                await connection.send_text(message)


def get_connection_manager():
    return ConnectionManager.get_instance()


class Timer(Singleton):
    def __init__(self):
        self.start_time: dict[str, float] = {}
        self.elapsed_time = {}

    def start(self, id: str):
        self.start_time[id] = perf_counter()

    def log(self, id: str, callback: Optional[Callable] = None):
        if id in self.start_time:
            elapsed_time = perf_counter() - self.start_time[id]
            del self.start_time[id]
            if id in self.elapsed_time:
                self.elapsed_time[id].append(elapsed_time)
            else:
                self.elapsed_time[id] = [elapsed_time]
            if callback:
                callback()

    def report(self):
        for id, t in self.elapsed_time.items():
            self.logger.info(
                f"{id:<30s}: {sum(t)/len(t):.3f}s [{min(t):.3f}s - {max(t):.3f}s] "
                f"({len(t)} samples)"
            )

    def reset(self):
        self.start_time = {}
        self.elapsed_time = {}


def get_timer() -> Timer:
    return Timer.get_instance()


def timed(func):
    if asyncio.iscoroutinefunction(func):
        async def async_wrapper(*args, **kwargs):
            timer = get_timer()
            timer.start(func.__qualname__)
            result = await func(*args, **kwargs)
            timer.log(func.__qualname__)
            return result
        return async_wrapper
    else:
        def sync_wrapper(*args, **kwargs):
            timer = get_timer()
            timer.start(func.__qualname__)
            result = func(*args, **kwargs)
            timer.log(func.__qualname__)
            return result
        return sync_wrapper

def split_sentences(streamed_output, punctuation_list):
    if not punctuation_list:
        return [streamed_output]
    # 将流式输出连接成一个字符串
    text = ''.join(streamed_output)
    pattern = r'(?<=[{}])'.format(punctuation_list)

    # 使用正则表达式按标点符号切割文本成句子
    sentences = re.split(pattern, text)

    # 去除空字符串和空格
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences

def checkFullSentence(input: str, punctuation_list):
    # 当句子结尾为标点符号时，回复True,表示INPUT为一句完整的句子。 否则则为False.
    pattern = "[" + punctuation_list + "]$"
    if not punctuation_list:
        return False
    if input:
        if len(input) < 1:
            return False
        elif not re.search(pattern, input[-1]):
            return False
        else:
            return True
    return False

def contains_only_punctuation(sentence, punctuation_list):
    # 当句子只包含标点符号时，回复True, 否则则为False.
    pattern = r'^[^\w\s]+$'
    match = re.match(pattern, sentence)
    if match:
        return True
    else:
        return False