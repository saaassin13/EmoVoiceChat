# 情绪语音聊天助手 (Emo-Voice-Chat)

一个基于深度学习的本地离线语音聊天系统，支持情绪识别、上下文理解和情绪匹配的语音回复。完全开源，适配 NVIDIA RTX 3090 单卡部署。

## ✨ 核心特性

- 🎤 **语音输入**：支持麦克风实时录音和音频文件上传
- 🗣️ **语音转文字 (STT)**：基于 SenseVoice 的高精度中文语音识别
- 😊 **情绪识别 (SER)**：自动识别用户语音中的情绪（开心/愤怒/悲伤/中性）
- 💬 **智能对话 (LLM)**：基于 Qwen2.5-7B 的上下文理解，根据情绪调整回复语气
- 🔊 **情绪语音合成 (TTS)**：根据回复情绪生成匹配语气的语音输出
- 📱 **Web 界面**：现代化的 Web 前端，支持移动端访问
- 🔄 **对话历史**：自动维护多用户对话上下文

## 🏗️ 系统架构

```
用户语音输入
    ↓
[STT + SER] 并行处理
    ↓
文本 + 情绪标签
    ↓
[LLM] 结合历史生成回复
    ↓
[TTS] 情绪匹配语音合成
    ↓
语音输出
```

### 核心模块

| 模块 | 功能 | 技术栈 |
|------|------|--------|
| **STT** | 语音转文字 | SenseVoice (FunASR) |
| **SER** | 情绪识别 | SenseVoice 内置情绪识别 |
| **LLM** | 对话生成 | Qwen2.5-7B-Instruct (4-bit量化) |
| **TTS** | 语音合成 | IndexTTS / Coqui TTS |
| **Web服务** | API接口 | FastAPI + Uvicorn |
| **前端** | 用户界面 | HTML5 + JavaScript |

## 🚀 快速开始

### 环境要求

- **操作系统**：Ubuntu 22.04+ (推荐)
- **GPU**：NVIDIA RTX 3090 (24GB 显存) 或同等配置
- **Python**：3.9+
- **CUDA**：11.8+ / 12.0+
- **显存占用**：约 14GB (所有模型加载后)

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd emo-voice-chat
```

2. **创建虚拟环境**
```bash
python3.9 -m venv venv
source venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **下载模型**

项目使用的模型会自动从 Hugging Face 或 ModelScope 下载，首次运行时会自动下载。

主要模型：
- **SenseVoice**：自动从 ModelScope 下载
- **Qwen2.5-7B-Instruct**：自动从 Hugging Face 下载
- **IndexTTS**：需要单独部署服务（见操作手册）

5. **启动服务**

**方式一：Web 服务（推荐）**
```bash
uvicorn web_service:app --host 0.0.0.0 --port 8001
```

然后在浏览器访问：`http://localhost:8001`

**方式二：命令行交互**
```bash
python -m src.app.main
```

## 📖 使用说明

### Web 界面使用

1. 打开浏览器访问 `http://localhost:8001`
2. 点击"开始录音"按钮，说话后点击"停止录音"
3. 系统会自动：
   - 识别你的语音内容
   - 分析你的情绪
   - 生成智能回复
   - 合成语音并播放

### API 接口

#### POST `/api/voice-chat`

上传语音文件进行对话。

**请求参数：**
- `file`: 音频文件 (multipart/form-data)
- `user_id`: 用户ID (可选，不提供则自动生成)

**响应示例：**
```json
{
  "user_id": "user_001",
  "audio_base64": "base64编码的音频数据",
  "user_text": "你好",
  "user_emotion": "neutral",
  "assistant_reply": "你好呀！有什么可以帮你的吗？",
  "assistant_emotion": "happy"
}
```

#### DELETE `/api/clear-history`

清除指定用户的对话历史。

**请求参数：**
- `user_id`: 用户ID (必填)

## 📁 项目结构

```
emo-voice-chat/
├── src/                    # 源代码目录
│   ├── app/               # 应用主流程
│   │   ├── main.py       # 命令行入口
│   │   └── pipeline.py   # 核心处理流程
│   ├── stt/              # 语音转文字模块
│   ├── ser/              # 情绪识别模块
│   ├── llm/              # 大语言模型模块
│   ├── tts/              # 语音合成模块
│   ├── context/          # 对话历史管理
│   ├── audio/            # 音频处理工具
│   └── config/           # 配置管理
├── data/                 # 数据目录
│   ├── history/         # 对话历史文件
│   ├── input/           # 输入音频
│   ├── output_voice/    # TTS输出音频
│   └── web_input/      # Web上传的音频
├── web_service.py       # FastAPI Web服务
├── index.html          # Web前端界面
├── requirements.txt    # Python依赖
└── README.md          # 本文档
```

详细的项目结构说明请参考 [项目结构说明.md](./项目结构说明.md)

## ⚙️ 配置说明

### 环境变量

可以通过环境变量自定义配置：

```bash
# STT 配置
export SENSEVOICE_MODEL_DIR="/path/to/model"
export SENSEVOICE_DEVICE="cuda:0"

# LLM 配置
export LLM_MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
export LLM_LOAD_IN_4BIT="true"

# TTS 配置
export TTS_BACKEND="indextts"  # 或 "coqui", "vits"
export INDEXTTS_SERVICE_URL="http://localhost:8000/tts"
```

详细配置说明请参考 [操作手册.md](./操作手册.md)

## 🔧 技术栈

- **深度学习框架**：PyTorch 2.4.0
- **STT**：FunASR (SenseVoice)
- **LLM**：Transformers + BitsAndBytes (4-bit量化)
- **TTS**：IndexTTS / Coqui TTS
- **Web框架**：FastAPI 0.128.0
- **音频处理**：PyAudio, SoundFile, Librosa

完整的技术栈说明请参考 [技术栈文档.md](./技术栈文档.md)

## 📊 性能指标

- **STT 识别速度**：60秒音频约 2-3秒 (RTX 3090)
- **LLM 生成速度**：约 20-30 tokens/秒 (4-bit量化)
- **TTS 合成速度**：10秒语音约 500ms
- **总延迟**：单轮对话约 3-5秒 (端到端)

## 🐛 常见问题

### 1. 显存不足 (OOM)

**问题**：运行时出现 CUDA out of memory 错误

**解决方案**：
- 使用更小的模型（如 Qwen2.5-3B）
- 降低量化位数（如使用 8-bit 而非 4-bit）
- 关闭其他占用显存的程序
- 使用 llama.cpp 后端（CPU推理）

### 2. 模型下载失败

**问题**：首次运行时模型下载失败

**解决方案**：
- 配置 Hugging Face 镜像站
- 使用 ModelScope 镜像（国内用户）
- 手动下载模型到本地

### 3. IndexTTS 服务连接失败

**问题**：TTS 合成失败，提示连接错误

**解决方案**：
- 确认 IndexTTS 服务已启动
- 检查 `INDEXTTS_SERVICE_URL` 配置是否正确
- 参考操作手册部署 IndexTTS 服务

更多问题请参考 [操作手册.md](./操作手册.md) 的"故障排除"章节。

## 📝 开发计划

- [ ] 支持更多情绪类型（惊讶、焦虑等）
- [ ] 优化显存占用，支持更小的GPU
- [ ] 添加语音克隆功能
- [ ] 支持多语言对话
- [ ] 添加对话导出功能

## 📄 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 🙏 致谢

- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - 语音识别
- [Qwen](https://github.com/QwenLM/Qwen) - 大语言模型
- [Coqui TTS](https://github.com/coqui-ai/TTS) - 语音合成
- [FastAPI](https://fastapi.tiangolo.com/) - Web框架

## 📧 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

---

**注意**：本项目仅供学习和研究使用，请勿用于商业用途。

