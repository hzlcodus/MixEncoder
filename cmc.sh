#!/bin/bash
#SBATCH --job-name=cmc       # Submit a job named "example"
#SBATCH --nodes=1                            # Using 1 node
#SBATCH --gres=gpu:1                          # Using 1 gpu
#SBATCH --time=0-48:00:00                     # 1 hour timelimit
#SBATCH --mem=10000MB                         # Using 10GB CPU Memory
#SBATCH --cpus-per-task=4                     # Using 4 maximum processor
#SBATCH --output=/home/cme/MixEncoder/slurm_output/slurm-%j.out

#python -u nlp_main.py -d ubuntu --model_class QAMatchModel --num_train_epochs 5 --label_num 0 -n 0 --val_batch_size 2 --train_batch_size 64  --pretrained_bert_path bert-base-uncased
#python -u nlp_main.py -d hard_msmarco  --load_model --load_model_path "/data/cme/mix/last_model/no_reg_v1.1-CMCModel_hard_msmarco" --model_save_prefix "no_reg_v1-" --model_class CMCModel --num_train_epochs 1 --composition pooler -n 1 --label_num 0 --train_batch_size 8 --step_max_query_num 8  --val_batch_size 2 --pretrained_bert_path Luyu/co-condenser-marco-retriever
python -u nlp_main.py -d hard_msmarco --model_save_prefix "no_reg_v1-" --model_class PolyEncoder --load_model --load_model_path "/data/cme/mix/last_model/no_reg_v1.1-PolyEncoder_hard_msmarco" -n 0 --num_train_epochs 3 --label_num 0 --train_batch_size 8 --val_batch_size 2 --one_stage --context_num 16 --pretrained_bert_path Luyu/co-condenser-marco-retriever
#python -u nlp_main.py -d hard_msmarco --model_class MatchDeformer -n 0 --num_train_epochs 3 --label_num 1 --top_layer_num 3 --val_batch_size 2 --train_batch_size 8 --one_stage --no_apex --seed 100
#python -u nlp_main.py -d hard_msmarco --model_save_prefix "no_reg_v1-" --model_class MatchParallelEncoder --num_train_epochs 3 -n 0 --label_num 0  --train_batch_size 8 --step_max_query_num 8 --val_batch_size 2  --one_stage --context_num 16 --no_apex --pretrained_bert_path Luyu/co-condenser-marco-retriever
