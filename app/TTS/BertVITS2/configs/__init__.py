from app.TTS.BertVITS2.configs.config import Config
import argparse
import sys,os

current_path = os.path.dirname(__file__)
sys.path.append(current_path)

parser = argparse.ArgumentParser()
# 在config文件里保存默认路径
default_config_path = os.path.join(current_path,"config.yml")
default_model_path = os.path.join(current_path.rsplit("\\",maxsplit = 1)[0],'bert')


# 为避免与以前的config.json起冲突，将其更名如下
parser.add_argument("-y", "--yml_config", type=str, default=default_config_path)
args, _ = parser.parse_known_args()
config = Config(args.yml_config)

jp_deberta_v2_path = os.path.join(default_model_path,"deberta-v2-large-japanese")
jp_bert_base_v3_path = os.path.join(default_model_path,"bert-base-japanese-v3")
jp_bert_large_v2_path = os.path.join(default_model_path,"bert-large-japanese-v2")
zh_roberta_large_path = os.path.join(default_model_path,"chinese-roberta-wwm-ext-large")