from transformers import pipeline
from datasets import load_dataset
from tqdm.auto import tqdm
import jsonlines
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import os


from eval import Classifier

OUTPUT_DIR = f"prompt/fewshot/v1/"


def filter_max_sents(text, k = 3):
    text = text.replace("\\n", "\n")
    sents = sent_tokenize(text)
    sents = sents[:k] if len(sents) > k else sents

    return " ".join(sents)


def generate_sentiment(num_shots: int = 30):
    dataset = load_dataset("SetFit/yelp_review_full", split="train").to_pandas()
    # topics = dataset.label.unique()
    topics = [4, 0]

    for topic in tqdm(topics, position=1):
        subset = dataset[dataset.label == topic].sample(num_shots, random_state=42)
        if topic == 4:
            topic = "positive"
        else:
            topic = "negative"
            
        prompt = f"""These are {topic} reviews.\n\n===\n""" + "\n===\n".join(map(filter_max_sents, subset.text)) + "\n===\n"
        
        with open(f"{OUTPUT_DIR}/{topic}-{num_shots}.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

def generate_emotion(num_shots: int = 30):
    emotion_dataset = load_dataset("SetFit/emotion", split="train").to_pandas()
    topics = emotion_dataset.label_text.unique()
    clf_toxicity = Classifier("s-nlp/roberta_toxicity_classifier")
    emotion_dataset["toxicity"] = emotion_dataset.text.map(lambda x: clf_toxicity.classify_item(x, 1, True, True))
    # print(emotion_dataset.head())
    for topic in tqdm(topics, position=1):
        subset = emotion_dataset[(emotion_dataset.label_text == topic) & (emotion_dataset.toxicity <= 0.5)] #.sample(num_shots, random_state=42)
        subset["length"] = subset.text.str.len()
        # print(subset, len(subset))
        subset = subset[(subset.length >= 15) & (subset.length <= 64)]
        # print(subset, len(subset))
        # subset = subset.sort_values(by=["length"])
        # subset = subset.iloc[:num_shots, :]
        subset = subset.sample(num_shots, random_state=42)

        prompt = f"""These are text containing feelings of {topic}.\n\n===\n""" + "\n===\n".join(subset.text) + "\n===\n"
        # print(prompt)
        # print("--" * 10)

        with open(f"{OUTPUT_DIR}/{topic}-{num_shots}.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

def generate_bbc_news(num_shots: int = 10):
    news_dataset = load_dataset("SetFit/bbc-news", split="train").to_pandas()
    all_topics = news_dataset.label_text.unique()

    for topic in all_topics:
        topic = topic.lower()
        topic_dataset = news_dataset[news_dataset.label_text == topic].sample(num_shots, random_state=42)
        lines = []

        for i in range(num_shots):
            item = topic_dataset.iloc[i]
            sents = sent_tokenize(item["text"])
            sents = sents[:3] if len(sents) > 3 else sents

            lines.append(" ".join(sents).replace("  ", " "))

        prompt = f"These are news articles of {topic} topics\n\n" + "\n===\n".join(lines) + "\n===\n"
        with open(f"{OUTPUT_DIR}/{topic}-{num_shots}.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

def generate_news_category(num_shots: int = 10):
    news_dataset = load_dataset("heegyu/news-category-balanced-top10", split="train").to_pandas()
    all_topics = news_dataset.category.unique()

    for topic in all_topics:
        topic = topic.lower()
        topic_dataset = news_dataset[news_dataset.short_description.str.len() > 0][news_dataset.category == topic.upper()].sample(42, random_state=42)
        lines = []

        for i in range(num_shots):
            item = topic_dataset.iloc[i]
            # lines.append("{title}".format(
            #     title=item["headline"],
            #     desc=item["short_description"],
            # ))
            lines.append(item["headline"])

        prompt = f"Write news titles and short descriptions of {topic} topics\n\n" + "\n".join(lines) + "\n===\n"
        with open(f"{OUTPUT_DIR}/{topic}-{num_shots}.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

if __name__ == "__main__":
    OUTPUT_DIR = f"prompt/fewshot/v1"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    generate_sentiment(10)
    generate_emotion(30)
    generate_bbc_news(5)
