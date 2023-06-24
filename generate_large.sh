
# Determine how many texts to generate for each topic
N=3
PROMPT="prompt/fewshot/v1"
DIR="output/large-fewshot/test"
mkdir -p $DIR

DEXPERTS="models/DExperts-large/finetuned_gpt2_nontoxic"
DEXPERTS_ANTI="heegyu/DExperts-large/finetuned_gpt2_toxic"

GATE="no"
OMEGA=30
ALPHA=1.0
RANKING_SCOPE=10

run() {
    TYPE=$1

    if [ $GATE = "no" ]
    then
        OUTPUT="$DIR/$TYPE.jsonl"
    else
        OUTPUT="$DIR/$TYPE-gate-t005.jsonl"
    fi

    python run_fewshot.py --model "gpt2-large" --model-type "$TYPE" \
        --n $N \
        --toxic-model $DEXPERTS \
        --nontoxic-model $DEXPERTS_ANTI \
        --gate-model $GATE \
        --disc_weight $OMEGA \
        --alpha $ALPHA \
        --ranking_scope $RANKING_SCOPE \
        --gate-threshold 0.005 \
        --prompt-dir $PROMPT \
        $OUTPUT
}



run "gpt2"

# use gated detoxifier, 
GATE="s-nlp/roberta_toxicity_classifier"
run "gedi"
run "discup"
