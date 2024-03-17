import argparse
import configparser
from pathlib import Path
import os
from app.utils.utils import Singleton
class AppConfig(Singleton):
    def __init__(self):
        # 初始化命令行参数
        parser = argparse.ArgumentParser(description="Just an example",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("-cfg", "--config", help="config设置文件路径.", default="config.ini")
        args = parser.parse_args()

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
        self.asr_model_path = get_config_variable(config,['AsrSetting','asrModelPath'])
        self.asr_model_path = os.path.join(self.proj_dir,self.asr_model_path)
        self.use_gpu = get_config_variable(config, ['AsrSetting', 'use_gpu'])
        self.use_int8 = get_config_variable(config, ['AsrSetting', 'use_int8'])
        self.beam_size = get_config_variable(config, ['AsrSetting', 'beam_size'])
        self.num_workers = get_config_variable(config, ['AsrSetting', 'num_workers'])
        self.vad_filter = get_config_variable(config, ['AsrSetting', 'vad_filter'])
        self.local_files_only = get_config_variable(config, ['AsrSetting', 'local_files_only'])

        #LLMSetting
        self.SPARKAPIDomain_v3 = get_config_variable(config, ['LLMSetting', 'SPARKAPIDomain_v3'])
        self.sparkDomain_v3 = get_config_variable(config, ['LLMSetting', 'sparkDomain_v3'])
        self.SPARK_API_ID = get_config_variable(config, ['LLMSetting', 'SPARK_API_ID'])
        self.SPARK_API_KEY = get_config_variable(config, ['LLMSetting', 'SPARK_API_KEY'])
        self.SPARK_API_SECRET = get_config_variable(config, ['LLMSetting', 'SPARK_API_SECRET'])

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
