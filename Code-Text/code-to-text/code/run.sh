lang=java 
lr=5e-5
batch_size=32
beam_size=10
source_length=256
target_length=128
data_dir=../dataset
output_dir=pert_model/$lang
train_file=$data_dir/$lang/train_pert.jsonl
test_file=$data_dir/$lang/test_pert.jsonl
dev_file=$data_dir/$lang/valid.jsonl
epochs=10 
pretrained_model=microsoft/codebert-base
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

while getopts ":rie" opt
do
    case $opt in
        r)
            python3 run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs
            ;;
        i)
            python3 run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size
            ;;
        e)
            python3 ../evaluator/evaluator.py model/$lang/dev.gold < model/$lang/dev.output
            ;;
        ?)
            echo "there is unrecognized parameter."
            exit 1
            ;;
    esac
done
