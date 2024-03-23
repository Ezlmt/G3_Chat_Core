import argparse
import configparser
from pathlib import Path
import os
from app.utils.utils import Singleton
from app.utils.logger import get_logger

logger = get_logger(__name__)
class AppConfig(Singleton):
    def __init__(self):
     # 初始化命令行参数
        parser = argparse.ArgumentParser(description="Just an example",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("-n", "--name", help="角色英文名称.", default="shulaibao")
        parser.add_argument("-llm", "--llm", help="大语言模型名称.", default="Spark")
        parser.add_argument("-md", "--model", help="大语言模型的模型型号.", default="V3")
        parser.add_argument("-cfg", "--config", help="config设置文件路径.", default="config.ini")
        parser.add_argument("-sc", "--scFlag", help="是否开启角色切换功能flag.", action="store_true")
        args = parser.parse_args()
        self.CharName = args.name
        self.LLMName = args.llm
        self.LLMModelName = args.model
        self.SwitchCharFlag = args.scFlag
        self.memoryFlag = True
        project_dir = Path(__file__).resolve().parent.parent.parent
        self.proj_dir = str(project_dir)
        self.config_path = os.path.join(self.proj_dir,args.config)

    # 初始化config.ini中变量
        config = configparser.ConfigParser()
        self.config_files = config.read(self.config_path, 'utf-8')
        if len(self.config_files) == 0:
            raise FileNotFoundError('配置文件 config.ini 未找到，请检查是否配置正确！')

        # RouterSetting
        self.ipaddress = get_config_variable(config, ['RouterSetting', 'ipaddress'])
        self.asr_port = get_config_variable(config, ['RouterSetting', 'asrPort'])
        self.tts_port = get_config_variable(config, ['RouterSetting', 'ttsPort'])

    # AsrSetting
        self.asr_model_path = get_config_variable(config,['AsrSetting','asr_model_path'])
        # Faster-Whisper
        self.whisper_asr_model = get_config_variable(config,['AsrSetting','whisper_asr_model'])
        self.use_gpu = get_config_variable(config, ['AsrSetting', 'use_gpu'])
        self.use_int8 = get_config_variable(config, ['AsrSetting', 'use_int8'])
        self.beam_size = get_config_variable(config, ['AsrSetting', 'beam_size'])
        self.num_workers = get_config_variable(config, ['AsrSetting', 'num_workers'])
        self.vad_filter = get_config_variable(config, ['AsrSetting', 'vad_filter'])
        self.local_files_only = get_config_variable(config, ['AsrSetting', 'local_files_only'])

        # FunASR
        self.ali_asr_model = get_config_variable(config, ['AsrSetting', 'ali_asr_model'])
        self.ali_punc_model = get_config_variable(config, ['AsrSetting', 'ali_punc_model'])
        self.ali_vad_model = get_config_variable(config, ['AsrSetting', 'ali_vad_model'])
        self.ali_asr_online_model = get_config_variable(config, ['AsrSetting', 'ali_asr_online_model'])
        self.ali_model_revision = get_config_variable(config, ['AsrSetting', 'ali_model_revision'])
        self.ngpu = get_config_variable(config, ['AsrSetting', 'ngpu'])
        self.device = get_config_variable(config, ['AsrSetting', 'device'])
        self.ncpu = get_config_variable(config, ['AsrSetting', 'ncpu'])


    #CharacterSetting
        self.CharNameDic = get_config_variable(config, ['CharNameList'], default={}, return_type=dict)

    #LLMConfigs
        self.LLMNameDic = get_config_variable(config, ['LLM'], default={}, return_type=dict)
        self.LLMModelNameDic = get_config_variable(config, ['MODEL'], default={}, return_type=dict)
        # SPARK LLM CONFIGS
        if self.LLMName not in self.LLMNameDic:
            self.LLMName = "Spark"
        if self.LLMModelName not in self.LLMModelNameDic:
            self.LLMModelName = "V3"

    #LLMSetting
        self.SPARKAPIDomain_v2 = get_config_variable(config,['LLMSetting','SPARKAPIDomain_v2'])
        self.SPARKAPIDomain_v3 = get_config_variable(config, ['LLMSetting', 'SPARKAPIDomain_v3'])
        self.sparkDomain_v2 = get_config_variable(config, ['LLMSetting', 'sparkDomain_v2'])
        self.sparkDomain_v3 = get_config_variable(config, ['LLMSetting', 'sparkDomain_v3'])
        self.SPARK_API_ID = get_config_variable(config, ['LLMSetting', 'SPARK_API_ID'])
        self.SPARK_API_KEY = get_config_variable(config, ['LLMSetting', 'SPARK_API_KEY'])
        self.SPARK_API_SECRET = get_config_variable(config, ['LLMSetting', 'SPARK_API_SECRET'])


    #Const data
        self.VECTOR_SEARCH_TOP_K = get_config_variable(config, ['const_data', 'TOP_K'], default=1, return_type=int)
        self.MEMORY_LENGTH = get_config_variable(config, ['const_data', 'MemoryLength'], default=1, return_type=int)
        self.ConversationSplitSTR = get_config_variable(config, ['const_data', 'ConversationSplitSTR'], default='$\t$')
        self.local_embed_model = get_config_variable(config, ['const_data', 'local_embed_model'])
        self.Model_DIR = get_config_variable(config, ['const_data', 'Embed_Model_Dir'])

    #PromptTemplate
        self.DataRootPath = get_config_variable(config, ['PromptTemplate', 'DataRootPath'])
        self.CharDataPath = get_config_variable(config, ['PromptTemplate', 'CharDataPath'])
        self.TemplatePath = get_config_variable(config, ['PromptTemplate', 'TemplatePath'])
        self.memoryTemplate = get_config_variable(config, ['PromptTemplate', 'memoryTemplate'])
        self.noMemoryTemplate = get_config_variable(config, ['PromptTemplate', 'noMemoryTemplate'])
        self.ChatPresetTemplate = get_config_variable(config, ['PromptTemplate', 'ChatPresetTemplate'])
        self.EmbedTemplate = get_config_variable(config, ['PromptTemplate', 'EmbedTemplate'])
        self.LLMTemplate = get_config_variable(config, ['PromptTemplate', 'LLMTemplate'])


        def validate_config(self):
            attributes = vars(self)  # 获取类实例的所有属性和值
            for attr, value in attributes.items():
                logger.debug(f"{attr} = {value}")

def get_config_instance() -> AppConfig:
    return AppConfig.get_instance()



def get_config_variable(config, keys, default='', return_type=str):
    try:
        value = config
        for key in keys:
            value = value[key]
        value = return_type(value)
        print(f"{keys}: {value}")
    except ValueError:
        print(f"{keys} Error: {ValueError}")
        value = default
    return value
