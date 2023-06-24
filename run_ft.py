import os
import click
import jsonlines, json
import pandas as pd
from pathlib import Path
from typing import Optional, List, Iterable, Dict, Any

import torch
from torch import FloatTensor, LongTensor
from tqdm import tqdm
import jsonlines
from itertools import chain


ALLOWED_MODELS = ["gpt3", "gpt2", "dexperts", "pplm", "gedi", "discup"]

FINETUNED_MODELS = {
    "bbc-news": "./models/gpt2-small/gpt2-bbc-news",
    "emotion": "./models/gpt2-small/gpt2-emotion",
    "sentiment": "./models/gpt2-small/gpt2-yelp-polarity",
}

ALL_TOPICS = {
    "sentiment": ["negative", "positive"],
    "emotion": ["sadness", "joy", "love", "anger", "fear", "surprise"],
    "bbc-news": ["business", "entertainment", "politics", "sport", "tech"],
}


torch.set_grad_enabled(False)


@click.command()
@click.argument("output-file")
@click.option("--model-type", required=True, type=click.Choice(ALLOWED_MODELS))
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
@click.option("--alpha", default=0.5, help="Hyperparameter for dexperts")
@click.option(
    "--filter_p",
    default=0.8,
    type=float,
    help="Hyperparameter for truncation of p_base",
)
@click.option(
    "--gate-threshold",
    default=0.5,
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
@click.option(
    "--top_p", default=1.0, type=float, help="Hyperparameter for nucleus sampling"
)
@click.option(
    "--top_k", default=50, type=int, help="Hyperparameter for nucleus sampling"
)
@click.option("--fp16/--no-fp16", default=False, type=bool, help="float16")
def main(
    output_file: str,
    model_type: str,
    nontoxic_model: str,
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
    gate_threshold: float,
    filter_p: float,
    target_p: float,
    fp16: bool,
    top_p: float,
    top_k: int,
):
    assert resume or overwrite or not os.path.exists(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    fout = jsonlines.open(output_file, "a" if resume else "w")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    progress = tqdm(total=n * sum(map(len, ALL_TOPICS.values())), desc=output_file)

    config = {
        "top_p": top_p,
        "top_k": top_k,
        "n": n,
        "batch_size": batch_size,
        "model": "ft-models",
        "model_type": model_type,
        "max_tokens": max_tokens,
        "toxic_model": toxic_model,
        "nontoxic_model": nontoxic_model,
        "gate_model": gate_model,
        "prompt": list(chain(ALL_TOPICS.values())),
        "alpha": alpha,
        "float16": fp16,
    }
    with open(output_file + ".config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    for topic, model in FINETUNED_MODELS.items():
        progress.desc = f"{topic} - {model}"
        prompts = ALL_TOPICS[topic]

        if model_type == "gpt2":
            from generator.gpt2 import GPT2Generator

            generator = GPT2Generator(
                model_name=model,
                num_return_sequences=batch_size,
                max_tokens=max_tokens,
                p=top_p,
                float16=fp16,
                device=device,
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
                p=top_p,
                gate_threshold=gate_threshold,
                float16=fp16,
                device=device,
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
                top_k=top_k,
                top_p=top_p,
                gate_model_name=gate_model,
                float16=fp16,
                gate_threshold=gate_threshold,
                device=device,
            )
        elif model_type == "pplm":
            pass

        if topic == "bbc-news":
            generator.max_tokens = 128
        elif topic == "emotion":
            generator.max_tokens = 64
        else:
            generator.max_tokens = 32

        for prompt in prompts:
            prompt = f"topic: {prompt}\n"
            for _ in range(n):
                # print(prompt)
                gens = generator.generate(prompt)
                for g in gens:
                    fout.write({"model_type": model_type, "prompt": prompt, "text": g})

                progress.update(1)

        del generator

    fout.close()


if __name__ == "__main__":
    main()
