from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
model_path = "./results/final_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
text = "This movie was amazing!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
probs = torch.softmax(outputs.logits, dim=1)
print(f"Positive sentiment: {probs[0][1]:.2%}")