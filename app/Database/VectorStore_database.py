import os
from langchain_xfyun.document_loaders import UnstructuredFileLoader
from langchain_xfyun.vectorstores import FAISS
import torch
from langchain_xfyun.embeddings.huggingface import HuggingFaceEmbeddings
import sentence_transformers
from sklearn.metrics.pairwise import cosine_similarity
from app.utils.logger import get_logger
from pathlib import Path
from app.configs.base_configs import AppConfig

logger = get_logger(__name__)

class embedManager():
    def __init__(self, local_embed_model, config_instance: AppConfig):
        DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.embeddings = HuggingFaceEmbeddings(model_name=local_embed_model)
        self.embeddings.client = sentence_transformers.SentenceTransformer(self.embeddings.model_name, device=DEVICE)
        self.config_ins = config_instance

    def validate_embeddings(self):
        return self.embeddings.model_name

    def update_vector_store(self, filepath:str):
        """ EMBED向量化filepath文件数据，返回向量数据库 """

        logger.info('开始 update_vector_store......')
        if not os.path.exists(filepath):
            logger.warning("路径不存在")
            return None
        elif os.path.isfile(filepath):
            file = os.path.split(filepath)[-1]
            try:
                refact_txt_spliter(filepath) # 将 \r 全部替换成 \n
                loader = UnstructuredFileLoader(filepath, mode="elements")
                docs = loader.load()
                logger.info(f"文件存在，{file} 已成功加载")
            except Exception as error:
                logger.warning(f"文件存在，{file} 未能成功加载")
                logger.warning(f"An error occurred:{error}")
                return None

        elif os.path.isdir(filepath):
            docs = []
            for file in os.listdir(filepath):
                fullfilepath = os.path.join(filepath, file)
                try:
                    loader = UnstructuredFileLoader(fullfilepath, mode="elements")
                    docs += loader.load()
                    logger.info(f"路径存在，{file} 已成功加载")
                except:
                    logger.warning(f"路径存在，{file} 未能成功加载")

        vector_store = FAISS.from_documents(docs, self.embeddings)
        return vector_store

    def get_candidate_questions(self, filepath):
        """
        根据splitSTR分割filepath文件，然后将问题部分向量化.
         最后分别列表化：问题（candidate_questions），回答（candidate_answer），EMBED向量化candidate_questions；
        """
        logger.info('开始get_candidate_questions......')
        doc = filepath
        docDic = {}
        splitSTR = self.config_ins.ConversationSplitSTR  # 分割用的字符
        if os.path.exists(filepath):
            logger.info(f"文件存在，{filepath} 成功加载")
            with open(doc, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for index, l in enumerate(lines):
                    if l.strip() == "":
                        continue
                    logger.debug(f'l : {l}')
                    try:
                        [q, a] = l.strip('\n').split(splitSTR)
                    except Exception as e:
                        logger.warning(f"[错误信息] 当前行缺少 ‘{splitSTR}’ 作为分隔符，当前行的内容是{l} ")
                        continue
                    qa = {q: a}
                    docDic.update(qa)
            self.candidate_questions = list(docDic.keys())
            self.candidate_answers = list(docDic.values())
            self.candidate_questions_embeddings = [self.embeddings.embed_query(answer) for answer in self.candidate_questions]

    def get_embeddings(self, query: str):
        """
        计算query与问题列表中各个项目的相似度，返回最接近的答案。
        阈值为0.8.
        """
        logger.info('开始检索conversation信息，寻找匹配回答......')
        question_embedding = self.embeddings.embed_query(query)
        similarity_scores = cosine_similarity([question_embedding], self.candidate_questions_embeddings)[0]
        top_index = similarity_scores.argsort()[-1:][0]
        answer = self.candidate_answers[top_index]
        question = self.candidate_questions[top_index]
        logger.info(f"搜索结果：{question}, 相似值：{round(similarity_scores[top_index],2)}")
        if round(similarity_scores[top_index],2) > 0.8:
            return answer
        else:
            logger.info('没有适配的搜索结果。')
            return ""

def log_docs_from_similarity_search(docs):
    for document in docs:
        page_content = document.page_content if hasattr(document, 'page_content') else ''
        if page_content != "":
            logger.info(f"source_documents: {page_content}")

# 将 \r 全部替换成 \n
def spliter_replace(text:str):
    import re
    compressed_text = re.sub(r"\r+", "\n", text)
    return compressed_text

# 打开文件，把文件内容里的 \r 全部替换成 \n
def refact_txt_spliter(filepath):
    with open(filepath,'r', encoding='utf-8') as f:
        text = f.read()
    with open(filepath,'w', encoding='utf-8') as f:
        f.write(spliter_replace(text))


# 测试embedding
if __name__ == '__main__':
    proj_dir = str(Path(__file__).resolve().parent.parent.parent)
    char = "shulaibao"
    path = proj_dir + "\\app\\Database\\models\\text2vec-large-chinese"
    knowledgePath = proj_dir + f"\\data\\character\\{char}\\{char}_knowledge.txt"
    conversationPath = proj_dir + f"\\data\\character\\{char}\\{char}_conversation.txt"
    config_instance = AppConfig.get_instance()
    e = embedManager(path, config_instance)
    logger.info("validate_embeddings: " + e.validate_embeddings())
    vs = e.update_vector_store(knowledgePath)


    #
    # e.get_candidate_questions(conversationPath)
    #
    # while True:
    #     query = input("\n用户：")
    #     if query.strip() == "stop":
    #         break
    #     elif query.strip() == "":
    #         continue
    #     answer = e.get_embeddings(query)
    #     if (len(answer)>0):
    #         result = answer
    #         logger.info(result)
    #     else:
    #         result = vs.similarity_search(query, k=config_instance.VECTOR_SEARCH_TOP_K)
    #         log_docs_from_similarity_search(result)

