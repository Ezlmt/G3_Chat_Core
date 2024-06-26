# 欢迎来到 G3-ChatBot-SHULaiBao

<div style="text-align:center">
   <img src="./Resources/shulaibao.jpg" width="300" height="300">
</div>

# 🤗项目介绍
本项目基于**Paraformer**、**Bert-VITS2**、**星火大模型**等模型实现了一套针对**虚拟角色交互系统**的服务端开发工作。默认使用角色名称为**鼠来宝**。
支持内容：

- 通过**Modelscope**调用达摩院**Paraformer**，如需提高识别精度，可在`config.ini`中修改模型名称或在`app\ASR\Paraformer\funasr_wss_server.py`，中修改识别流程。
- 更换声音模型，参考下方**配置修改**部分流程。

**本项目涉及模型均可通过Modelscope平台自行下载，但声音模型可自行添加。通过整合包安装则默认携带两个声音模型。**


# 👉快速启动


<a name="MsAF8"></a>
## 一、整合包


1. 通过以下链接下载整合包内容：`https://pan.baidu.com/s/1_kg6O9k_5eNLZpoNZYSxnQ?pwd=2x6b`
2. 解压整合包
3. 通过项目根目录的`.bat`文件启动服务
<a name="PtdTi"></a>
## 二、git下载


1. 通过`Pycharm`或`Conda`创建虚拟环境
2. 执行`pip install -r requirements.txt`，下载所需包文件
3. 在终端中执行`xxx_setup.py`文件，以启动对应的服务
<a name="LyWuy"></a>
# 🤔配置修改


- 有关**本项目的主要配置信息**保存在根目录的`config.ini`文件中，如**模型名称**、**文件路径**、**路由设置**等。（ip地址在`_setup.py`中自动获取）
- 关于**声音模型的配置信息**，主要存在放`app\TTS\BertVITS2v202\config.yml`文件的`server`键值对下，包含了模型启动的声音模型信息。
   - 声音模型文件保存在`app\TTS\BertVITS2v202\Data\meimei\models`目录下，并以`G_xxx.pth`的格式进行存放。（请务必注意**以G开头**！否则将无法识别。）
   - 声音模型主要是基于 **Bert-VITS2V2.0.2** 版本整合包
   - 训练教程：[本地训练,立等可取,30秒音频素材复刻霉霉讲中文音色基于Bert-VITS2V2.0.2_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1VG411i7U3/?spm_id_from=333.788&vd_source=955a84fe268292d6c91c5cb596f4feb7)
<a name="L34qX"></a>
## 一、角色创建

1. 在`data\character`目录下创建**角色名称文件夹**，如`shulaibao`（可以复制修改自其他角色文件夹）
2. 文件夹需包含`_conversation.txt`、`_knowledge.txt`、`_orders.json`、`_prompt.txt`
   1. `conversation.txt`文件需要注意**分隔符**！可以从其他语句复制，修改两端($)内容
   2. `knowledge.txt`文件中知识条目以**回车**进行分割
   3. `_prompt.txt`主要用于设置角色的人设
   4. `_orders.json`用于设置指令,如需有,请按照**领域细分**的方式进行设置
