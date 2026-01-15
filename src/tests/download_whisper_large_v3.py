import whisper

model = whisper.load_model("large-v3")  # 第一次会下载
print("模型加载完成:", type(model))