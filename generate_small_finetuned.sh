OUTPUT_DIR="output/test"

mkdir -p $OUTPUT_DIR

# for DExperts-small
# DEXPERTS="models/DExperts-small/gpt2-non-toxic"
# DEXPERTS_ANTI="models/DExperts-small/gpt2-toxic"

# for DExperts-lage
DEXPERTS_ANTI="models/DExperts-large/finetuned_gpt2_toxic"
DEXPERTS="models/DExperts-large/finetuned_gpt2_nontoxic"

GATE="no"
OMEGA=30
ALPHA=1.0
ITER_NUM=20
GATE_THRESHOLD=0.005
N=3

run() {
    TYPE=$1
    if [ $GATE = "no" ]
    then
        OUTPUT="$TYPE"
    else
        OUTPUT="$TYPE-gate-t005"
    fi
    echo $OUTPUT

    python run_ft.py \
        --model-type "$TYPE" \
        --toxic-model $DEXPERTS \
        --nontoxic-model $DEXPERTS_ANTI \
        --gate-model $GATE \
        --disc_weight $OMEGA \
        --alpha $ALPHA \
        --gate-threshold $GATE_THRESHOLD \
        --n $N \
        --overwrite \
        $OUTPUT_DIR/$OUTPUT.jsonl    
}

# run "gpt2"
# run "gedi"
# run "dexperts"

GATE="s-nlp/roberta_toxicity_classifier"
# run "gedi"
run "dexperts"

