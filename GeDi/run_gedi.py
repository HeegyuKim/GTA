import pickle
from pathlib import Path
from typing import Optional, List, Iterable, Dict, Any
import json
from typing import TypeVar, Iterable, List, Union, Any
import numpy as np
import torch
from tqdm.auto import tqdm
import os
import click
import jsonlines
import pandas as pd

from modeling_gpt2 import GPT2LMHeadModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from transformers import (
    GPT2Config,
    GPT2Tokenizer
)

T = TypeVar('T')


ALLOWED_MODELS = ['gpt3', 'gpt2', 'dexperts', 'dexperts-gpt3', 'pplm']
ALLOWED_PROMPT = ["yelp", "emotion", "bbc-news"]
PROMPT = {
    "yelp": ["topic: positive\n", "topic: negative\n"],
    "emotion": [f"topic: {k}\n" for k in ["sadness", "joy", "love", "anger", "fear", "surprise"]],
    "bbc-news": [f"topic: {k}\n" for k in ["tech", "business", "sport", "entertainment", "politics"]],
}

torch.set_grad_enabled(False)



def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
        
def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch

def make_generations_col(generations, responses):
    for generation, response in zip(generations, responses):
        # if response['response']:
        #     response = unpack_scores(response['response'])[0]
        # else:
        #     response = {x: None for x in PERSPECTIVE_API_ATTRIBUTES_LOWER}
        yield {'text': generation, **response}


def collate(dataset: Optional[pd.DataFrame], generations: List[str], responses: Iterable[Dict[str, Any]], output_file: str):
    generations_col_iter = make_generations_col(generations, responses)
    if dataset is None:
        generations_col = list(tqdm(generations_col_iter, total=len(generations), desc='Collating files'))
        dataset = pd.DataFrame(generations_col)
    else:
        assert len(generations) % len(dataset) == 0
        n = len(generations) // len(dataset)
        print(f"Detected samples per prompt:", n)
        generations_col = list(tqdm(batchify(generations_col_iter, n), total=len(dataset), desc='Collating files'))
        dataset['generations'] = generations_col

    dataset.to_json(output_file, orient='records', lines=True)




@click.command()
@click.argument('output-file')
@click.option('--prompt', required=True, type=click.Choice(ALLOWED_PROMPT))
@click.option('--use-eos/--use-dataset', default=False, help='Whether to use EOS or a dataset file for generation.')
@click.option('--model', required=True, help='Equivalent to `model_name_or_path` in transformers.')
@click.option('--classifier-model', type=str, default=None, help='Classifier for Gated Detoxifier')
@click.option('--n', default=25, help='Number of samples to generate for each prompt. When used with --eos')
@click.option('--max-tokens', default=32, help='Number of tokens (usually BPE) to generate for each prompt.')
@click.option('--batch-size', default=8)
@click.option('--resume/--no-resume', default=False)
@click.option('--overwrite/--no-overwrite', default=False)
@click.option('--gedi/--no-gedi', default=True)
@click.option('--filter_p', default=0.8, type=float, help='1 - rho from paper, should be between 0 and 1 higher filter_p means more aggressive topic steering')
@click.option('--target_p', default=0.8, type=float, help='tau from paper, preserves tokens that are classified as correct topic')
@click.option('--disc_weight', default=30, type=int, help='omega from paper, higher disc_weight means more aggressive topic steering (30)')
@click.option('--top_p', default=1.0, type=float, help='Hyperparameter for nucleus sampling')
@click.option('--top_k', default=50, type=int, help='Hyperparameter for top-k sampling')
@click.option('--logits_scale', default=10.0, type=float, help='GeDi logits scale')
def main(output_file: str, prompt: str, use_eos: bool, model: str, 
         n: int, max_tokens: int, batch_size: int, 
         resume: bool, overwrite: bool, gedi: bool,
         disc_weight: float, logits_scale: float,
         classifier_model: str, filter_p: float, target_p: float, 
         top_p: float, top_k: int):
    
    assert resume or overwrite or not os.path.exists(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained(model)
    model = GPT2LMHeadModel.from_pretrained(model)#, load_in_half_prec=True)
    model = model.to(device).eval()

    if gedi:
        gedi_model_name_or_path = '../models/GeDi/cc_lm_detox'
        gedi_model = GPT2LMHeadModel.from_pretrained(gedi_model_name_or_path).eval().to(device)

    if classifier_model is not None and classifier_model != "no":
        classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_model)
        classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_model).eval().to(device)
    else:
        classifier_model, classifier_tokenizer = None, None


    # disc_weight = 30 # #omega from paper, higher disc_weight means more aggressive topic steering (30)
    # filter_p = 0.8 #1 - rho from paper, should be between 0 and 1 higher filter_p means more aggressive topic steering
    # target_p = 0.8 #tau from paper, preserves tokens that are classified as correct topic
    class_bias = 0 #hyperparameter that determines class prior, set to uniform by default
    code_desired = "dirty"
    code_undesired = "clean"

    def generate_text(prompt):
        text_ids = tokenizer.encode(prompt)
        encoded_prompts=torch.LongTensor(text_ids).unsqueeze(0).to(device)
        
        attr_class = 1

        generated_sequence = model.generate(
            input_ids=encoded_prompts,
            pad_lens=None,
            max_length=min(1024, encoded_prompts.shape[1] + max_tokens),
            # min_length=min(1024, encoded_prompts.shape[1] + max_tokens),
            top_k=top_k,
            top_p=top_p,
            # repetition_penalty= 1.2,
            # rep_penalty_scale= 10,
            eos_token_ids = [50256],
            pad_token_id = 50256,
            do_sample= True,
            penalize_cond= True,
            gedi_model= gedi_model if gedi else None,
            tokenizer= tokenizer,
            disc_weight= disc_weight,
            filter_p = filter_p,
            target_p = target_p,
            class_bias = class_bias,
            attr_class = attr_class,
            code_0 = code_desired,
            code_1 = code_undesired,
            multi_code=None,
            num_return_sequences=batch_size,
            classifier_model=classifier_model, 
            classifier_tokenizer=classifier_tokenizer
            )

        texts = [tokenizer.decode(output, skip_special_tokens=True)[len(prompt):] for output in generated_sequence.tolist()[0]]
        # print(generated_sequence.tolist())
        # texts = [tokenizer.decode(output, skip_special_tokens=True) for output in generated_sequence.tolist()[0]]
        return texts

    prompts = PROMPT[prompt]
    fout = jsonlines.open(output_file, "a" if resume else "w")

    progress = tqdm(total=len(prompts) * n, desc=output_file)

    for prompt in prompts:
        for _ in range(n):
            gens = generate_text(prompt)
            for g in gens:
                fout.write({
                    "prompt": prompt,
                    "generation": g
                })

            progress.update(1)
    
    fout.close()

if __name__ == "__main__":
    main()