# character_manager.py

from app.configs.base_configs import AppConfig
import os
from app.Database.VectorStore_database import embedManager
from app.utils.utils import Singleton
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CharacterManager(Singleton):
    def __init__(self, config: AppConfig, character_name):
        self.app_config = config
        self.llmModelName = self.app_config.LLMModelName
        self.CharNameDic = config.CharNameDic
        self.llm_name = self.app_config.LLMName
        self.splitSTR=self.app_config.ConversationSplitSTR
        self.proj_dir = config.proj_dir
        self.memoryFlag = config.memoryFlag

        # PromptTemplate 绝对路径
        self.TemplateFilePath = os.path.join(self.proj_dir, self.app_config.DataRootPath, self.app_config.TemplatePath)

        # character 绝对路径
        self.CharDataPath = os.path.join(self.app_config.proj_dir,self.app_config.DataRootPath,self.app_config.CharDataPath)
        self.embedding = embedManager(os.path.join(config.proj_dir,config.Model_DIR, config.local_embed_model), self.app_config)
        self.SEARCH_TOP_K = config.VECTOR_SEARCH_TOP_K
        self.updateChar(character_name)

    def charDataFileList(self):
        """
        根据self.character_name，生成当前角色的数据文件名。
        """
        self.conversation_file = f"{self.character_name}_conversation.txt"
        self.knowledge_file = f"{self.character_name}_knowledge.txt"
        self.prompt_file = f"{self.character_name}_prompt.txt"
        self.orders_file = f"{self.character_name}_orders.json"
        self.OrderFilePath = os.path.join(self.CharDataPath, self.character_name, self.orders_file)
    def update_templates(self):
        """
        根据当前角色设置，更新templates需要文件的路径。
        """
        # 是否使用带有Memory Template的Prompt
        if self.memoryFlag:
            self.memoryTemplateStr = self.loadfile(os.path.join(self.TemplateFilePath, self.app_config.memoryTemplate))
        else:
            self.memoryTemplateStr = self.loadfile(os.path.join(self.TemplateFilePath, self.app_config.noMemoryTemplate))
        logger.debug("self.TemplateFilePath: " + self.TemplateFilePath)
        logger.debug("self.app_config.ChatPresetTemplate: "+ self.app_config.ChatPresetTemplate)
        # 确定角色名称，Prompt
        self.ChatPresetTemplateStr = self.loadfile(os.path.join(self.TemplateFilePath, self.app_config.ChatPresetTemplate))
        logger.debug("self.ChatPresetTemplateStr: "+ self.ChatPresetTemplateStr)
        # 确定角色回复，Prompt
        self.EmbedTemplateStr = self.loadfile(os.path.join(self.TemplateFilePath, self.app_config.EmbedTemplate))
        self.LLMTemplateStr = self.loadfile(os.path.join(self.TemplateFilePath, self.app_config.LLMTemplate))
        self.promptPath=os.path.join(self.CharDataPath, self.character_name, self.prompt_file)
        self.LLM_prompt_contents = self.loadfile(self.promptPath)
        logger.debug(f"prompt的路径是：{self.CharDataPath}")
        logger.debug(f"prompt的内容是：{self.LLM_prompt_contents}")

    def knowledge_prompt_template_build(self):
        """
        生成knowledge_prompt_templates文本。
        """
        self.knowledge_prompt_template = self.ChatPresetTemplateStr + self.LLM_prompt_contents + self.LLMTemplateStr + self.memoryTemplateStr

    def conversation_prompt_template_build(self):
        """
        生成conversation_prompt_templates文本。
        """
        self.conversation_prompt_template = self.ChatPresetTemplateStr + self.LLM_prompt_contents + self.EmbedTemplateStr + self.memoryTemplateStr


    def build_knowledge(self):
        """
        更新knowledge数据库。
        """
        self.knowledgePath = os.path.join(self.CharDataPath, self.character_name,self.knowledge_file)
        self.vs = self.embedding.update_vector_store(self.knowledgePath)

    def build_conversation(self):
        """
        更新conversation数据库。
        """
        self.conversationPath = os.path.join(self.CharDataPath, self.character_name,self.conversation_file)
        self.embedding.get_candidate_questions(self.conversationPath)

    def search_knowledge(self, query):
        """
        查找knowledge数据库中SEARCH_TOP_K个数据条目。
        """
        return self.vs.similarity_search(query, k=self.SEARCH_TOP_K)

    def updateChar(self, CharName):
        """
        切换当前角色为CharName，更新所有对应该角色的数据。
        """
        if CharName not in self.CharNameDic:
            logger.error(f"invalid '{CharName}'. fail to switch Character!")
        else:
            self.character_name = CharName
            self.CharChineseName = self.CharNameDic[CharName]
            self.charDataFileList()
            self.update_templates()
            self.knowledge_prompt_template_build()
            self.conversation_prompt_template_build()
            self.build_knowledge()
            self.build_conversation()
            print("charater 更新完毕！")

    def loadfile(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as prompt_file:
                result = prompt_file.read()
            logger.info(f"成功加载{filepath}")
            return result
        else:
            logger.warning(f"无法发现{filepath}")
            return None

    def savefile(self, filepath, file_content):
        if os.path.exists(filepath):
            os.remove(filepath)
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(file_content)
            logger.info(f'[result]:{filepath} Updated Successfully!\n')
        else:
            logger.warning(f"无法发现{filepath}")

    def validate_config(self):
        attributes = vars(self)  # 获取类实例的所有属性和值
        for attr, value in attributes.items():
            logger.info(f"{attr} = {value}")

def get_character_instance() -> CharacterManager:
    return CharacterManager.get_instance()

if __name__ == '__main__':
    manager = CharacterManager.get_instance()
