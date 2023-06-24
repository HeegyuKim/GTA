gpu=0
temperature=1.0
batch_size=1
file_name='../eval'
target_type="positive"
model_name_or_path="/home2/zhanghanqing/pretrained_model/gpt2/large"
data_path="../datasets/nontoxic_prompts-10k.jsonl"

embedding_checkpoint="../check_point/detoxic/distill_tuning_positive_(5,5).ckpt"
template="(5,5)"
beta=0.999
tuning_name="distill_tuning"
task_name="detoxic"
seed=2


mode="ctg"
iter_num=20
top_p=1.0
CUDA_VISIBLE_DEVICES=$gpu

for ranking_scope in 10
do 
    echo  ---ranking_scope--$ranking_scope---------
    
       python ../main.py  --batch_size $batch_size --ranking_scope $ranking_scope --file_name $file_name  --target_type $target_type --model_name_or_path $model_name_or_path  --embedding_checkpoint $embedding_checkpoint --iter_num $iter_num --top_p $top_p --beta $beta --template $template --tuning_name $tuning_name --task_name $task_name --seed $seed --data_path $data_path
    wait

done


