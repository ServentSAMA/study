import whisper

model = whisper.load_model("small", "cuda")

result = model.transcribe("E:\\PR\\FC2-PPV-2569870.mp3")

print(result["text"])
