import time

from fastapi import Request
import requests
import json
from pydub import AudioSegment
from pydub.utils import make_chunks
import asyncio
import aiohttp
async def fetch_audio(url:str, segment:bytes):
    async with aiohttp.ClientSession() as session:
        audio = {"audio": segment}
        async with session.post(url, data=audio) as response:
            if response.status == 200:
                try:
                    result_json = await response.json()
                    print(result_json)
                    # 处理解析后的数据
                except json.JSONDecodeError as e:
                    print("JSON解析错误:", e)
            else:
                print("请求错误，状态码:", response.status)


async def main():
    url = "http://127.0.0.1:7840/recognition"
    audio_path = 'test.wav'
    # audio = AudioSegment.from_file(audio_path, "wav")
    # size = 3500  # 切割的毫秒数 60s=60000
    # chunks = make_chunks(audio, size)  # 将文件切割为60s一个
    # for i,chunk in enumerate(chunks):
    #     chunks[i] = chunks[i].export(format='wav').read()
    # for chunk in chunks:
    #     print(requests.post(url=url,files = {"audio":chunk}).json())

    STATUS_FIRST_FRAME = 0  # 第一帧的标识
    STATUS_CONTINUE_FRAME = 1  # 中间帧标识
    STATUS_LAST_FRAME = 2  # 最后一帧的标识

    frame_size = 20000
    intervel = 0.04
    status = STATUS_FIRST_FRAME  # 初始状态为第一帧

    with open(audio_path,"rb") as f:
        while True:
            buf = f.read(frame_size)
            if not buf:
                status = STATUS_LAST_FRAME
            if status == STATUS_FIRST_FRAME:
                print(requests.post(url=url,files = {"audio":buf}).json())
                status = STATUS_CONTINUE_FRAME
            elif status == STATUS_CONTINUE_FRAME:
                print(requests.post(url=url, files={"audio": buf}))
            elif status == STATUS_LAST_FRAME:
                print(requests.post(url=url, files={"audio": buf}))
                time.sleep(1)
                break
            break
            time.sleep(intervel)
if __name__ == '__main__':
    asyncio.run(main())
    # url = "http://127.0.0.1:7840/recognition"
    # with open("test.wav","rb") as file:
    #     audio = file.read()
    # audio = {"audio":audio}
    # result = requests.post(url=url,files=audio,stream=True)
    # for item in result.iter_content(chunk_size=1024):
    #     item_str = item.decode(encoding="UTF-8")
    #     try:
    #         item_str = item_str.replace("\x00",'') # 将字节字符串解码为字符串
    #         print(item_str)
    #         # print(item_str.replace("\x00",''))
    #         item_json = json.loads(item_str)  # 解析字符串为JSON对象
    #         result_content = item_json["result"]  # 提取result的值
    #         print(result_content)
    #     except:
    #         print(item_str)
    #         continue
