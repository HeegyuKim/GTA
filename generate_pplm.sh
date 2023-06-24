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