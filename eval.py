from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
import torch
import jsonlines
from tqdm.auto import tqdm
from collections import defaultdict
import click
import re
import pandas as pd
import numpy as np
from googleapiclient import discovery
import json
from typing import List
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Classifier:
    def __init__(self, model_name) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name)
            .to(device)
            .eval()
        )

    @torch.no_grad()
    def classify_item(self, text, label_index, clean=False, return_prob: bool = False):
        # if clean:
        #     text = clean_text(text)

        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}

        logits = self.model(**tokens).logits.softmax(-1)
        if return_prob:
            preds = logits[0].cpu().tolist()
            return preds[label_index]
        else:
            return logits[0].cpu().argmax(-1).item() == label_index


class PerspectiveAPI:
    def __init__(self) -> None:
        API_KEY = os.environ["PERSPECTIVE_API_KEY"]
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=API_KEY,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )
        self.api = self.client.comments()

    def __call__(self, text: str):
        analyze_request = {
            "comment": {"text": text},
            "languages": ["en"],
            "requestedAttributes": {"TOXICITY": {}},
        }

        response = self.api.analyze(body=analyze_request).execute()
        score = response["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
        # print(json.dumps(response, indent=2))
        return score


class PerplexityEstimator:
    def __init__(self, model_name: str = "gpt2") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def classify_item(self, text, clean=False):
        tokens = self.tokenizer([text], return_tensors="pt").to(device)["input_ids"]
        if tokens.shape[1] == 0:
            return 0

        loss = self.model(input_ids=tokens, labels=tokens).loss
        return loss.cpu().item()


class TopicPPL:
    def __init__(self) -> None:
        self.ppls = {
            "sentiment": PerplexityEstimator("heegyu/gpt2-yelp-polarity"),
            "emotion": PerplexityEstimator("heegyu/gpt2-emotion"),
            "news": PerplexityEstimator("heegyu/gpt2-bbc-news"),
        }

    @torch.no_grad()
    def classify_item(self, topic, text, clean=False):
        return self.ppls[topic2group[topic]].classify_item(text, clean)


def score_accuracy(items, preds, use_prompt_label=True, true_label=None):
    total = defaultdict(lambda: 0)
    correct = defaultdict(lambda: 0)

    for item, p in zip(items, preds):
        label = item["topic"]
        total[label] += 1
        # print(label, p)
        if use_prompt_label:
            if label == p:
                correct[label] += 1
        elif p == true_label:
            correct[label] += 1
    # print(items)
    # print(preds)
    # print(list(correct.keys()), list(total.keys()))
    for k in total.keys():
        # print(k, correct[k] / total[k] ,f"{correct[k]} of {total[k]}")
        correct[k] = correct.get(k, 0) / total[k]

    return correct


topic_category = {
    "sentiment": ["negative", "positive"],
    "emotion": ["sadness", "joy", "love", "anger", "fear", "surprise"],
    # "news": [x.lower() for x in [
    #     "ENTERTAINMENT", "POLITICS", "WELLNESS", "TRAVEL", "STYLE & beauty",
    #     "PARENTING", "HEALTHY living", "QUEER VOICES", "FOOD & DRINK", "BUSINESS",
    #     ]],
    "news": ["business", "entertainment", "politics", "sport", "tech"],
}

topic2group = {v: g for g, l in topic_category.items() for v in l}


@click.command()
@click.option("--perspective/--no-perspective", default=True, type=bool, help="use perspective api or classifier to evaluate toxicity")
@click.option("--large", default=False, type=bool, help="is it large experiment result? if true, use gpt2-large for evaluating perplexity.")
@click.argument("files", nargs=-1)
def main(
    perspective: bool,
    large: bool,
    files: List[str]
    ):
    files = tqdm(files)
    for file in files:
        files.desc = file
        handle_file(perspective=perspective, large=large, filename=file)


def get_topic(x):
    return x.split("\n", 1)[0].replace("topic: ", "").strip().split("-", 1)[0]


def clean_text(x):
    if not x:
        return None

    if x[0] == "=":
        x = x[1:]

    x = x.split("==", 1)[0]
    # x = re.sub(r"[^a-zA-Z\"'\!\?\s\.\,\&\(\)\-]", "", x)
    x = x.strip()

    return x


def handle_file(
    perspective: bool,
    large: bool,
    filename: str
    ):

    df = pd.read_json(filename, lines=True)
    if "topic" in df.columns:
        df["topic"] = df.topic.map(lambda x: x.split("-", 1)[0])
    if "prompt" in df.columns:
        df["topic"] = df.prompt.map(get_topic)
    else:
        df["prompt"] = ""

    df.rename(columns={"generation": "text"}, inplace=True)
    df.text = df.text.map(clean_text)
    df = df[df.text.str.len() > 0]

    print("filtered: ", len(df))

    # clf_news = Classifier("heegyu/roberta-base-news-category-top10")
    clf_news = Classifier("Umesh/distilbert-bbc-news-classification")
    clf_sentiment = Classifier("VictorSanh/roberta-base-finetuned-yelp-polarity")
    clf_emotion = Classifier("bhadresh-savani/bert-base-uncased-emotion")
    topic_classifier = {
        "sentiment": clf_sentiment,
        "emotion": clf_emotion,
        "news": clf_news,
    }

    def eval_topic_text(topic, prompt, text):
        group = topic2group[topic]
        index = topic_category[group].index(topic)
        # text = prompt.split("\n", 1)[1] + text
        return topic_classifier[group].classify_item(text, index, False, False)

    df["topic_accuracy"] = list(
        tqdm(
            map(
                lambda x: eval_topic_text(x[0], x[1], x[2]),
                zip(df.topic, df.prompt, df.text),
            ),
            desc="evaluating topics",
        )
    )
    del topic_classifier
    del clf_news
    del clf_sentiment
    del clf_emotion


    # evaluate toxicity
    if perspective:
        clf_toxicity = PerspectiveAPI()
        df["toxicity"] = df.text.map(clf_toxicity)
    else:
        clf_toxicity = Classifier("s-nlp/roberta_toxicity_classifier")
        df["toxicity"] = df.text.map(lambda x: clf_toxicity.classify_item(x, 1, True, True))
    del clf_toxicity


    clf_grammar = Classifier("cointegrated/roberta-large-cola-krishna2020")
    df["grammar"] = df.text.map(lambda x: clf_grammar.classify_item(x, 0, True, True))
    del clf_grammar

    if large:
        # for fewshot generation
        ppl = PerplexityEstimator("gpt2-large")
        df["loss"] = list(tqdm(
            map(lambda x: ppl.classify_item(x[2]), zip(df.topic, df.prompt, df.text)),
            desc="evaluating ppls"
            ))
    else:
        ppl = TopicPPL()
        df["loss"] = list(
            tqdm(
                map(
                    lambda x: ppl.classify_item(x[0], x[1] + x[2]),
                    zip(df.topic, df.prompt, df.text),
                ),
                desc="evaluating topic ppls",
            )
        )

    del ppl


    # save result
    loss2ppl(df).to_csv(filename + ".eval.csv")
    loss2ppl(df.groupby("topic").mean()).to_csv(filename + ".eval_topic_mean.csv")
    loss2ppl(df.mean()).to_csv(filename + ".eval_mean.csv")


def loss2ppl(df):
    df["ppl"] = np.exp(df.loss)
    return df


if __name__ == "__main__":
    main()
