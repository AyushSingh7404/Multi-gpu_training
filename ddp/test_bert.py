import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_path="./saved_bert_model"):
    """Load the trained model"""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    return model, tokenizer, device

def predict_sentiment(text, model, tokenizer, device):
    """Predict sentiment for a single text"""
    # Tokenize
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(outputs.logits, dim=-1)
    
    # Convert to readable format
    label = "positive" if prediction.item() == 1 else "negative"
    confidence = probabilities.max().item()
    
    return label, confidence

def main():
    """Simple inference test"""
    print("=== BERT Sentiment Analysis ===\n")
    
    # Load model
    model, tokenizer, device = load_model()
    
    # Test sentences
    test_texts = [
        "I absolutely love this movie! It's amazing!",
        "This is the worst film I've ever seen.",
        "The product works okay, nothing special.",
        "Fantastic experience, highly recommend!",
        "Terrible service, very disappointed."
    ]
    
    print("Testing model predictions:\n")
    print("-" * 60)
    
    for text in test_texts:
        label, confidence = predict_sentiment(text, model, tokenizer, device)
        
        print(f"Text: {text}")
        print(f"Prediction: {label.upper()}")
        print(f"Confidence: {confidence:.3f}")
        print("-" * 60)
    
    # Interactive mode
    print("\n=== Interactive Mode ===")
    print("Enter text to analyze (type 'quit' to exit):\n")
    
    while True:
        user_input = input("Your text: ").strip()
        
        if user_input.lower() == 'quit':
            break
            
        if user_input:
            label, confidence = predict_sentiment(user_input, model, tokenizer, device)
            print(f"→ {label.upper()} (confidence: {confidence:.3f})\n")

if __name__ == "__main__":
    main()