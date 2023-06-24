PROMPT=emotion-5
MODEL=heegyu/gpt2-emotion
OUTPUT_DIR=test-generations/$PROMPT-2
PROMPTS_DATASET=../DExperts/prompts/$PROMPT.jsonl

python run_gedi.py \
    --use-dataset \
    --dataset-file $PROMPTS_DATASET \
    --model $MODEL \
    --no-gedi \
    --n 1 \
    "$OUTPUT_DIR/no-gedi"
