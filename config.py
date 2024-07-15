import numpy as np
import random
import torch
import argparse
from dataBaseSet.database import Database
import datetime

RAW_DATASET_ROOT_FOLDER = 'data'
EXPERIMENT_ROOT = 'experiments'
STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'
PROJECT_NAME = 'GammaRecForClothes'


def set_template(args):
    database=Database(host=args.host,user=args.user,password=args.password,db_name=args.dbname,port=args.port)
    args.dataset_code=datetime.datetime.now().strftime("%Y %b %d")

    if torch.cuda.is_available(): args.device = 'cuda'
    else: args.device = 'cpu'
    args.optimizer = 'AdamW'
    args.lr = 0.001
    args.weight_decay = 0.01
    args.enable_lr_schedule = False
    args.decay_step = 10000
    args.gamma = 1.
    args.enable_lr_warmup = False
    args.warmup_steps = 100

    args.metric_ks = [1, 5, 10, 20, 50]
    args.rerank_metric_ks = [1, 5, 10]
    args.best_metric = 'Recall@10'
    args.rerank_best_metric = 'NDCG@10'

    return database


parser = argparse.ArgumentParser()

################
# Dataset
################
parser.add_argument('--dataset_code', type=str, default=None)
parser.add_argument('--host', type=str, default=None)
parser.add_argument('--user', type=str, default=None)
parser.add_argument('--password', type=str, default=None)
parser.add_argument('--dbname', type=str, default=None)
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--min_rating', type=int, default=0)
parser.add_argument('--min_uc', type=int, default=6)
parser.add_argument('--min_sc', type=int, default=6)
parser.add_argument('--seed', type=int, default=42)


################
# Dataloader
################
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--sliding_window_size', type=float, default=1.0)
parser.add_argument('--negative_sample_size', type=int, default=10)

################
# Trainer
################
# optimization #
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--optimizer', type=str, default='AdamW', choices=['AdamW', 'Adam'])
parser.add_argument('--weight_decay', type=float, default=None)
parser.add_argument('--adam_epsilon', type=float, default=1e-9)
parser.add_argument('--momentum', type=float, default=None)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--max_grad_norm', type=float, default=5.0)
parser.add_argument('--enable_lr_schedule', type=bool, default=True)
parser.add_argument('--decay_step', type=int, default=10000)
parser.add_argument('--gamma', type=float, default=1)
parser.add_argument('--enable_lr_warmup', type=bool, default=True)
parser.add_argument('--warmup_steps', type=int, default=100)

# evaluation #
parser.add_argument('--val_strategy', type=str, default='iteration', choices=['epoch', 'iteration'])
parser.add_argument('--val_iterations', type=int, default=100)  # only for iteration val_strategy
parser.add_argument('--early_stopping', type=bool, default=True)
parser.add_argument('--early_stopping_patience', type=int, default=20)
parser.add_argument('--metric_ks', nargs='+', type=int, default=[1, 5, 10, 20, 50])
parser.add_argument('--rerank_metric_ks', nargs='+', type=int, default=[1, 5, 10])
parser.add_argument('--best_metric', type=str, default='Recall@10')
parser.add_argument('--rerank_best_metric', type=str, default='NDCG@10')
parser.add_argument('--use_wandb', type=bool, default=False)

################
# Retriever Model
################
parser.add_argument('--model_code', type=str, default=None)
parser.add_argument('--bert_max_len', type=int, default=50)
parser.add_argument('--bert_hidden_units', type=int, default=64)
parser.add_argument('--bert_num_blocks', type=int, default=2)
parser.add_argument('--bert_num_heads', type=int, default=2)
parser.add_argument('--bert_head_size', type=int, default=32)
parser.add_argument('--bert_dropout', type=float, default=0.3)
parser.add_argument('--bert_attn_dropout', type=float, default=0.3)
parser.add_argument('--bert_mask_prob', type=float, default=0.25)

################
# LLM Model
################
"""
check the candidate of model name in huggingface
1. Undi95/Meta-Llama-3-8B-hf
2. NousResearch/Llama-2-7b-hf
3. unsloth/llama-2-7b-bnb-4bit
4. unsloth/gemma-2b-bnb-4bit
5. google/gemma-2-9b
6. google/gemma-2b
 we select 6th model because of our GPU conditions
"""

parser.add_argument('--llm_base_model', type=str, default="google/gemma-2b")
parser.add_argument('--llm_base_tokenizer', type=str, default="google/gemma-2b")
parser.add_argument('--hf_token', type=str, default=None)
parser.add_argument('--llm_max_title_len', type=int, default=16)
parser.add_argument('--llm_max_text_len', type=int, default=512) #1536
parser.add_argument('--llm_max_history', type=int, default=10)
parser.add_argument('--llm_train_on_inputs', type=bool, default=False)
parser.add_argument('--llm_negative_sample_size', type=int, default=9)  # 19 negative & 1 positive
parser.add_argument('--llm_system_template', type=str,  # instruction
    default="Given user style and history in survey, recommend an item from the candidate pool with its index letter.")
parser.add_argument('--llm_input_template', type=str, \
    default='User style:{}; \n User history: {}; \n Candidate pool: {}')
parser.add_argument('--llm_load_in_4bit', type=bool, default=True)
parser.add_argument('--llm_retrieved_path', type=str, default='./')
parser.add_argument('--llm_cache_dir', type=str, default=None)

################
# Lora
################
parser.add_argument('--lora_r', type=int, default=4)
parser.add_argument('--lora_alpha', type=int, default=16)
parser.add_argument('--lora_dropout', type=float, default=0.05)
parser.add_argument('--lora_target_modules', type=list, default=['q_proj', 'v_proj'])
parser.add_argument('--lora_num_epochs', type=int, default=1)
parser.add_argument('--lora_val_iterations', type=int, default=500)
parser.add_argument('--lora_early_stopping_patience', type=int, default=20)
parser.add_argument('--lora_lr', type=float, default=1e-4)
parser.add_argument('--lora_micro_batch_size', type=int, default=4)

################
# File paths for inference
################
parser.add_argument('--save_folder_path', type=str, default='./saveFiles')

args = parser.parse_args()
