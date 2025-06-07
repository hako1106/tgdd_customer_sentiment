import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class CommentDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def collate_batch(batch_texts, tokenizer):
    return tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True)


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def analyze_sentiment(input_path, output_path, model, tokenizer, device, labels):
    df = pd.read_csv(input_path)
    texts = df['comment_text_remove_emojis'].fillna("").tolist()

    dataset = CommentDataset(texts)
    dataloader = DataLoader(
        dataset, batch_size=16, collate_fn=lambda x: collate_batch(x, tokenizer)
    )

    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            all_preds.extend(preds.cpu().tolist())

    df['sentiment'] = [labels[p] for p in all_preds]
    df.to_csv(output_path, index=False)

    return df


def run_sentiment_analysis(
    input_path="data/processed/facebook_comments_cleaned.csv",
    output_path="data/processed/facebook_comments_cleaned_with_sentiment.csv",
    model_name="models/bert_sentiment_vietnamese"
):
    """Chạy phân tích cảm xúc trên bình luận Facebook"""
    print("\nRunning sentiment analysis...")
    labels = ['negative', 'neutral', 'positive']
    tokenizer, model, device = load_model(model_name)
    return analyze_sentiment(input_path, output_path, model, tokenizer, device, labels)


if __name__ == "__main__":
    df = run_sentiment_analysis()
