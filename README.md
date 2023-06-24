# Gated Detoxifier
This repository contains code for the paper "Gated Detoxifier"


# Dependencies
1. our code is based on `python 3.9`
1. Install required packages using pip
```
pip install -r requirements.txt
```
3. Download detoxifier models

| Method | Model Link |
---------| ---------------
| [PPLM](https://github.com/uber-research/PPLM) | hi |
| [GeDi](https://github.com/salesforce/GeDi) | [Detoxification CC-LM(355M)](https://storage.googleapis.com/sfr-gedi-data/gedi_detoxifier.zip) |
| [DExperts](https://github.com/alisawuffles/DExperts) | [Experts/Anti-experts(774M)](https://drive.google.com/uc?id=1HSrNMrq4OZ3nyTobNd2TZFcB5NYwluu-)  |
| [DisCup](https://github.com/littlehacker26/Discriminator-Cooperative-Unlikelihood-Prompt-Tuning) | [GPT2-large prompt embeddings](https://drive.google.com/file/d/1k4qSpYhuS1SYWL0SVmQ6CuSH_PdAYjdc/view)  |

# Text generation using detoxifier
If you want few-shot generation in large-scale LM, you need to specify prompt directory. There are already generated prompts in `prompt/fewshot/v1` directory. If you want another prompt, use code in `generate_prompt.py`.

The `run_ft.py` and `run_fewshot.py` code generates and stores N texts of 13 topics using detoxifier. There are 13 topics of three topic groups.
| Topic group | topics | dataset |
| -- | -- | -- |
| Sentiment | positive / negative | [yelp-polarity](https://huggingface.co/datasets/yelp_polarity) |
| Emotion | anger / fear / surprise / joy / sadness / love | [emotion](https://huggingface.co/datasets/dair-ai/emotion) |
| News | business / entertainment / politics / sport / tech | [bbc-news](https://huggingface.co/datasets/SetFit/bbc-news)

This example generate texts without detoxifier.

```bash
# small-scale fine-tuned model generation
python run_ft.py \
    --model-type "gpt2" \
    --n 1000 \
    $OUTPUT

# large-scale fewshot generation
python run_fewshot.py --model "gpt2-large" \
    --model-type "gpt2" \
    --n 100 \
    --prompt-dir prompt/fewshot/v1 \
    $OUTPUT
```

And you can change top-p and top-k argument, top-k is default 50, top-p is default 0.9(small) and 1.0(large).

## PPLM
For using PPLM, you need to make seperate environment. Setup a enviroment with `pip install -r requirements_pplm.txt`.
You cannot change parameters in argument. If you want to change other parameters, change [PPLM/pplm_generation.py](PPLM/pplm_generation.py).
```bash
cd PPLM

count=3

python3 pplm_gated.py \
    --top_k 50 \
    --top_p 0.9 \
    --print-result \
    --n $count \
    --label-class 1 \
    --sample \
    --gate_threshold 0.005 \
    --output-file "../output/test/pplm_gated_$count.jsonl"

python3 pplm.py \
    --top_k 50 \
    --top_p 0.9 \
    --print-result \
    --n $count \
    --label-class 1 \
    --sample \
    --output-file "../output/test/pplm_$count.jsonl"
```

## GeDi
Only you can change disc_weight(omega) in argument. If you want to change other parameters, change [GeDi/generator.py](GeDi/generator.py).
```
OMEGA=30

# small-scale
python run_ft.py \
    --model-type "gedi" \
    --disc_weight $OMEGA \
    --n $N \
    output/small/gedi.jsonl    

# large-scale
python run_fewshot.py \
    --model "gpt2-large" \
    --model-type "gedi" \
    --n 100 \
    --disc_weight $OMEGA \
    --prompt-dir prompt/fewshot/v1 \
    output/large/gedi.jsonl
```
## DExperts
```
ALPHA=1.0

# small-scale
python run_ft.py \
    --model-type "dexperts" \
    --alpha $ALPHA \
    --n 100 \
    output/small/dexperts.jsonl    

# large-scale fewshot
python run_fewshot.py \
    --model "gpt2-large" \
    --model-type "dexperts" \
    --n 100 \
    --alpha $ALPHA \
    --prompt-dir prompt/fewshot/v1 \
    output/large/dexperts.jsonl
```

## DisCup

Only large-scale generation is available for DisCup
```
# Gated Discup large
python run_fewshot.py \
    --model "gpt2-large" \
    --model-type "discup" \
    --n 100 \
    --ranking_scope 10 \
    --prompt-dir prompt/fewshot/v1 \
    output/large/discup_gated.jsonl
```

## Text generation using gated detoxifier
You can add a gate in any detoxifier. just add gate-model argument

```bash
GATE_MODEL="s-nlp/roberta_toxicity_classifier"
GATE_THRESHOLD=0.005

OMEGA=30

# Gated GeDi small
python run_ft.py \
    --model-type "gedi" \
    --disc_weight 30 \
    --n $N \
    --gate-model $GATE \
    --gate-threshold $GATE_THRESHOLD \
    output/small/gedi_gated.jsonl    

# gated-discup large
python run_fewshot.py \
    --model "gpt2-large" \
    --model-type "discup" \
    --n 100 \
    --ranking_scope 10 \
    --gate-model $GATE \
    --gate-threshold $GATE_THRESHOLD \
    --prompt-dir prompt/fewshot/v1 \
    output/large/discup.jsonl
```

# Evaluating generated texts
Receive your perspective api key from https://perspectiveapi.com/ or use classifier instead.

```bash
export PERSPECTIVE_API_KEY=your_api_key

# for small-scale
python eval.py output/small/gedi.jsonl

# for large-scale
python eval.py --large output/small/gedi.jsonl

# if you doesn't want to use perspective api for evaluating toxicity
python eval.py --no-perspective output/small/gedi.jsonl

```


### remove '\r' (For Windows)
```
sed -i 's/\r$//' ./generate_large.sh
```