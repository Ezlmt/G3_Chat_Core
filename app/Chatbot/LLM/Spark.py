from langchain_xfyun.chat_models import ChatSpark
from langchain_xfyun.chains import LLMChain
from langchain_xfyun.chains.question_answering import load_qa_chain
from langchain_xfyun.prompts import PromptTemplate
from langchain_xfyun.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_xfyun.callbacks.manager import CallbackManager
from app.Database.Character_Manager import CharacterManager
from langchain_xfyun.memory import ConversationBufferWindowMemory
from app.configs.base_configs import AppConfig,logger
import threading
from app.utils.utils import Singleton
import queue
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ThreadedGenerator:
    """
    可迭代数据存储类，用于流式回复的存放和使用。
    """
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration: raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)


class ChainStreamHandler(StreamingStdOutCallbackHandler):
    """
    重新实现on_llm_new_token， 来自langchain_xfyun StreamingStdOutCallbackHandler
    """
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        self.gen.send(token)

class SparkChatService(Singleton):
    def __init__(self, app_config:AppConfig, char:CharacterManager):
        self.app_config = app_config
        self.char = char
        self.updateModel(self.app_config.LLMModelName)
        self.memory = ConversationBufferWindowMemory(memory_key="chat_history", input_key="human_input", k=app_config.MEMORY_LENGTH)

    def run_chat(self, query):
        """
        非流式对话。
        ARGS:
            QUERY: STR，提问文本
        RETURN:
            resp: CHAIN类
        """
        g = ThreadedGenerator()
        answer = self.char.embedding.get_embeddings(query)
        if len(answer) > 0:
            resp = self.run_conversation_chat_thread(g, query, answer)
        else:
            resp = self.run_knowledge_chat_thread(g, query)
        return resp

    def run_chat_stream(self, query, ifstream=True):
        """
        流式对话。
        ARGS:
            QUERY: STR，提问文本
            ifstream默认为True
        RETURN:
            g: ThreadedGenerator类。
        """
        g = ThreadedGenerator()
        answer = self.char.embedding.get_embeddings(query)
        if len(answer) > 0:
            logger.info("找到合适的匹配结果，即将按照conversation内容进行回答....")
            threading.Thread(target=self.run_conversation_chat_thread, args=(g, query, answer, ifstream)).start()
        else:
            logger.info("未找到合适的匹配结果，开始寻找合适的knowledge条目内容....")
            threading.Thread(target=self.run_knowledge_chat_thread, args=(g, query, ifstream)).start()
        return g

    def run_knowledge_chat_thread(self, g, query, ifstream=False):
        """
        根据知识库进行回答。
        ARGS:
            g: ThreadedGenerator实例
            QUERY: STR，提问文本
        RETURN:
            resp: CHAIN类
        """
        try:
            LLM = ChatSpark(
                api_domain=self.sparkAPIDomain,
                spark_domain=self.sparkDomain,
                streaming=ifstream,
                callback_manager=CallbackManager([ChainStreamHandler(g)]),
                app_id=self.app_config.SPARK_API_ID,
                api_key=self.app_config.SPARK_API_KEY,
                api_secret=self.app_config.SPARK_API_SECRET,
            )
            docs = self.char.search_knowledge(query)

            if self.char.memoryFlag:
                prompt = PromptTemplate(
                    template=self.char.knowledge_prompt_template,
                    input_variables=["CharName", "chat_history", "context", "human_input"]
                )
                knowledge_chain = load_qa_chain(llm=LLM, chain_type="stuff", memory=self.memory, prompt=prompt, verbose=True)
                self.del_dupl_AIMessage()
            else:
                print(f"knowledge_prompt_template: {self.char.knowledge_prompt_template}")
                prompt = PromptTemplate(
                    template=self.char.knowledge_prompt_template,
                    input_variables=["CharName", "context", "human_input"]
                )
                knowledge_chain = load_qa_chain(llm=LLM, chain_type="stuff", prompt=prompt, verbose=True)

            return knowledge_chain({"input_documents": docs, "human_input": query, "CharName": self.char.CharChineseName},
                                   return_only_outputs=True)
        finally:
            g.close()

    def run_conversation_chat_thread(self, g, query, text, ifstream=False):
        """
        根据问答库进行回答。
        ARGS:
            g: ThreadedGenerator实例
            QUERY: STR，提问文本
            text：STR，问答库提取的匹配答案
        RETURN:
            resp: CHAIN类
        """
        try:
            LLM = ChatSpark(
                api_domain=self.sparkAPIDomain,
                spark_domain=self.sparkDomain,
                streaming=ifstream,
                callback_manager=CallbackManager([ChainStreamHandler(g)]),
                app_id=self.app_config.SPARK_API_ID,
                api_key=self.app_config.SPARK_API_KEY,
                api_secret=self.app_config.SPARK_API_SECRET,
            )

            if self.char.memoryFlag:
                prompt = PromptTemplate(
                    template=self.char.conversation_prompt_template,
                    input_variables=["CharName", "chat_history", "text", "human_input"]
                )

                conversation_chain = LLMChain(llm=LLM, prompt=prompt, verbose=True, memory=self.memory)
                self.del_dupl_AIMessage()               
            else:
                prompt = PromptTemplate(
                    template=self.char.conversation_prompt_template,
                    input_variables=["CharName", "text", "human_input"]
                )
                conversation_chain = LLMChain(llm=LLM, prompt=prompt, verbose=True)

            return conversation_chain({'human_input': query, 'text': text, "CharName":self.char.CharChineseName})
        finally:
            g.close()

    def del_dupl_AIMessage(self):
        # 如果记忆中有AI重复回复相同文本，则删除这条重复的记忆。
        unique_set = set()
        i = 1
        msgs = self.memory.chat_memory.messages
        while i <= len(msgs):
            if i % 2 == 0:
                curText = msgs[i - 1].content
                logger.debug(f"目前的AIMessage ID是：{i}，内容是: {curText}")
                if curText in unique_set:
                    logger.info(f"发现重复AIMessage: {curText}")
                    del msgs[i - 1]
                    del msgs[i - 2]
                unique_set.add(curText)
            i += 1

    def clearMemory(self):
        # 清除当前记忆
        if self.memory != None:
            self.memory.clear()
        else:
            logger.info(f"memory is None!")

    def updateModel(self, targetModel:str):
        # 更新LLM模型版本。
        if targetModel == "V3":
            self.sparkAPIDomain = self.app_config.SPARKAPIDomain_v3
            self.sparkDomain = self.app_config.sparkDomain_v3
            print(f"当前使用的模型为：SparkV3")
        elif targetModel == "V2":
            self.sparkAPIDomain = self.app_config.SPARKAPIDomain_v2
            self.sparkDomain = self.app_config.sparkDomain_v2
            print(f"当前使用的模型为：SparkV2")
        else:
            logger.warning(f"{targetModel}为非法LLM名称，切换失败。")


def get_spark_instance() -> SparkChatService:
    return SparkChatService.get_instance()

if __name__ == '__main__':
    AppConfig.initialize()
    config = AppConfig.get_instance()
    SparkChatService.initialize(config,"renzixi")
    sparkChat = SparkChatService.get_instance()
    CharacterManager.initialize(config,"renzixi")
    CharacterManager.get_instance()
    print(sparkChat)