import os
import click
import jsonlines, json
import pandas as pd
from pathlib import Path
from typing import Optional, List, Iterable, Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

import torch
from torch import FloatTensor, LongTensor
from tqdm import tqdm
from eval import clean_text
from transformers import pipeline, set_seed

ALLOWED_MODELS = ["gpt3", "gpt2", "dexperts", "pplm", "gedi", "discup"]
NEWS_TOPICS = [
    x.lower()
    for x in [
        "ENTERTAINMENT",
        "POLITICS",
        "WELLNESS",
        "TRAVEL",
        "STYLE & beauty",
        "PARENTING",
        "HEALTHY living",
        "QUEER VOICES",
        "FOOD & DRINK",
        "BUSINESS",
    ]
]


def read_prompts(prompt_dir):
    files = os.listdir(prompt_dir)
    prompts = {}

    for file in files:
        with open(f"{prompt_dir}/{file}", encoding="utf-8") as f:
            prompts[file.replace(".txt", "")] = f.read()

    return prompts


class ToxicityClassifier:
    def __init__(self, model_name, device, threshold: float) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name)
            .to(device)
            .eval()
        )
        self.device = device
        self.threshold = threshold

    @torch.no_grad()
    def __call__(self, text):
        # text = clean_text(text)

        tokens = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)

        logits = self.model(**tokens).logits.softmax(-1)[:, 1]
        outputs = (logits > self.threshold).float()
        # print(text, self.threshold)
        return outputs


torch.set_grad_enabled(False)


@click.command()
@click.argument("output-file")
@click.option(
    "--model", required=True, help="Equivalent to `model_name_or_path` in transformers."
)
@click.option("--model-type", required=True, type=click.Choice(ALLOWED_MODELS))
@click.option("--seed", default=42, help="random seed")
@click.option(
    "--prompt-dir", type=str, default="prompt/fewshot/v1", help="prompt file directory"
)
@click.option(
    "--toxic-model",
    type=str,
    default="DExperts/finetuned_gpt2_toxic",
    help="Anti-expert for DExperts",
)
@click.option(
    "--nontoxic-model",
    type=str,
    default="DExperts/finetuned_gpt2_nontoxic",
    help="Expert for DExperts",
)
@click.option(
    "--gate-model", type=str, default=None, help="Classifier for Gated Detoxifier"
)
@click.option(
    "--n",
    default=25,
    help="Number of samples to generate for each prompt. When used with --eos",
)
@click.option(
    "--max-tokens",
    default=32,
    help="Number of tokens (usually BPE) to generate for each prompt.",
)
@click.option("--batch-size", default=1)
@click.option("--resume/--no-resume", default=False)
@click.option("--overwrite/--no-overwrite", default=False)
@click.option(
    "--gate-threshold",
    default=0.5,
    type=float,
    help="Hyperparameter for truncation of p_base",
)
@click.option("--alpha", default=0.5, help="Hyperparameter for dexperts")
@click.option(
    "--filter_p",
    default=0.8,
    type=float,
    help="Hyperparameter for truncation of p_base",
)
@click.option(
    "--target_p",
    default=0.8,
    type=float,
    help="Hyperparameter for truncation of p_base",
)
@click.option("--disc_weight", default=15, type=float, help="GeDi omega")
@click.option("--logits_scale", default=10.0, type=float, help="GeDi logits scale")
@click.option("--ranking_scope", default=10, type=int, help="Discup ranking scope(top-k)")
@click.option(
    "--top_p", default=1.0, type=float, help="Hyperparameter for nucleus sampling"
)
@click.option(
    "--top_k", default=50, type=int, help="Hyperparameter for top-k sampling"
)
@click.option("--fp16/--no-fp16", default=False, type=bool, help="float16")
def main(
    output_file: str,
    model: str,
    model_type: str,
    nontoxic_model: str,
    prompt_dir: str,
    gate_threshold: float,
    seed: int,
    toxic_model: str,
    n: int,
    max_tokens: int,
    batch_size: int,
    resume: bool,
    overwrite: bool,
    disc_weight: float,
    logits_scale: float,
    gate_model: str,
    alpha: float,
    filter_p: float,
    target_p: float,
    fp16: bool,
    ranking_scope: int,
    top_p: float,
    top_k: int,
):
    set_seed(seed)
    assert resume or overwrite or not os.path.exists(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prompts = read_prompts(prompt_dir)

    if model_type == "gpt2":
        from generator.gpt2 import GPT2Generator

        generator = GPT2Generator(
            model_name=model,
            num_return_sequences=batch_size,
            max_tokens=max_tokens,
            p=top_p,
            device=device,
            float16=fp16,
        )
    elif model_type == "dexperts":
        from DExperts.dexperts import DExpertGenerator

        generator = DExpertGenerator(
            model_name=model,
            num_return_sequences=batch_size,
            max_tokens=max_tokens,
            expert_model_name=nontoxic_model,
            anti_expert_model_name=toxic_model,
            classifier_model_name=gate_model,
            alpha=alpha,
            device=device,
            p=top_p,
            float16=fp16,
        )
    elif model_type == "gedi":
        from GeDi.generator import GeDiGenerator

        generator = GeDiGenerator(
            model_name=model,
            num_return_sequences=batch_size,
            max_tokens=max_tokens,
            disc_weight=disc_weight,
            filter_p=filter_p,
            target_p=target_p,
            logits_scale=logits_scale,
            device=device,
            top_k=top_k,
            top_p=top_p,
            gate_model_name=gate_model,
            gate_threshold=gate_threshold,
            float16=fp16,
        )
    elif model_type == "discup":
        from DisCup.main import construct_generation_args
        from DisCup.control_generation import CTG

        args = construct_generation_args()
        generator = CTG(
            args,
            gate_model=ToxicityClassifier(
                gate_model, device=device, threshold=gate_threshold
            )
            if gate_model is not None and gate_model != "no"
            else None,
            max_tokens=max_tokens,
        )

    fout = jsonlines.open(output_file, "a" if resume else "w")

    progress = tqdm(total=len(prompts) * n, desc=output_file)

    config = {
        "top_p": top_p,
        "top_k": top_k,
        "ranking_scope": ranking_scope,
        "n": n,
        "batch_size": batch_size,
        "model": model,
        "model_type": model_type,
        "max_tokens": max_tokens,
        "toxic_model": toxic_model,
        "nontoxic_model": nontoxic_model,
        "gate_model": gate_model,
        "prompt": prompts,
        "alpha": alpha,
        "gate_threshold": gate_threshold,
        "float16": fp16,
    }
    with open(output_file + ".config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    for topic, prompt in prompts.items():
        if topic.split("-", 1)[0] in NEWS_TOPICS:
            generator.max_tokens = max_tokens * 4
        else:
            generator.max_tokens = max_tokens

        for _ in range(n):
            # print(prompt)
            gens = generator.generate(prompt)
            # print(gens)
            for g in gens:
                g = g.strip().split("===")[0].strip()

                if g.startswith("itle:"):
                    g = g.replace("itle:", "Title:")

                fout.write({"topic": topic, "text": g})
                print("generated!", topic, g)

            progress.update(1)

    fout.close()


if __name__ == "__main__":
    main()
