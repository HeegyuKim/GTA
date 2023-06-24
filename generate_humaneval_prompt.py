from transformers import pipeline
from datasets import load_dataset
from tqdm.auto import tqdm
import jsonlines
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import os


# nltk.download('punkt')
output_dir = "prompt/auto-eval"

def generate_sentiment(num_shots: int = 30):
    dataset = load_dataset("SetFit/yelp_review_full", split="train").to_pandas()
    # topics = dataset.label.unique()
    topics = [4, 0]

    with jsonlines.open(f"{output_dir}/yelp.txt", "w") as f:
        for topic in tqdm(topics, position=1):
            subset = dataset[dataset.label == topic].sample(num_shots, random_state=42)
            if topic == 4:
                topic = "positive"
            else:
                topic = "negative"
                
            for text in subset.text:
                text = sent_tokenize(text)[0]

                prompt = f"""topic: {topic}\n{text}. """
                f.write({"prompt": prompt})

def generate_emotion(num_shots: int = 30, prompt_tokens: int = 4):
    emotion_dataset = load_dataset("SetFit/emotion", split="train").to_pandas()
    topics = emotion_dataset.label_text.unique()

    with jsonlines.open(f"{output_dir}/emotion.txt", "w") as f:
        for topic in tqdm(topics, position=1):
            subset = emotion_dataset[emotion_dataset.label_text == topic].sample(num_shots, random_state=42)
            for text in subset.text:
                tokens = word_tokenize(text)
                text = text if len(tokens) <= prompt_tokens else " ".join(tokens[:prompt_tokens])
                prompt = f"""topic: {topic}\n{text}"""
                f.write({"prompt": prompt})

            if prompt_tokens == 0:
                prompt = f"""topic: {topic}\n"""
                for _ in range(num_shots):
                    f.write({"prompt": prompt})


def generate_news_category(num_shots: int = 10):
    news_dataset = load_dataset("SetFit/bbc-news", split="train").to_pandas()
    topics = news_dataset.label_text.unique()

    with jsonlines.open(f"{output_dir}/bbc-news.txt", "w") as f:
        for topic in tqdm(topics, position=1):
            subset = news_dataset[news_dataset.label_text == topic].sample(num_shots, random_state=42)
            for text in subset.text:
                text = sent_tokenize(text)[0]
                prompt = f"""topic: {topic}\n{text}"""
                f.write({"prompt": prompt})


if __name__ == "__main__":
    shots = 100
    output_dir = f"prompt/auto-eval-{shots}"
    
    os.makedirs(output_dir, exist_ok=True)
    generate_sentiment(shots)
    generate_emotion(shots, 0)
    generate_news_category(shots)