import os
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
from config import *
from LlamaRec.model import *
from dataloader import *
from trainer import *
from dataBaseSet.survey import SurveyDataset

from transformers import BitsAndBytesConfig
from pytorch_lightning import seed_everything
from modelStructure import GemmaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)


try:
    os.environ['WANDB_PROJECT'] = PROJECT_NAME
except:
    print('WANDB_PROJECT not available, please set it in config.py')

def main(args, database, export_root=None):
    seed_everything(args.seed)
    if export_root == None:
        export_root = EXPERIMENT_ROOT + '/' + args.llm_base_model.split('/')[-1] + '/' + args.dataset_code
    dataset=SurveyDataset(args,database)
    # # model
    # model, tokenizer = FastLanguageModel.from_pretrained(
    #     # model_name = "unsloth/llama-3-8b-bnb-4bit",
    #     model_name="unsloth/llama-3-8b-bnb-4bit",  # 모델 이름을 설정합니다.
    #     max_seq_length=args.llm_max_text_len,  # 최대 시퀀스 길이를 설정합니다.
    #     dtype=None,  # 데이터 타입을 설정합니다.
    #     load_in_4bit=args.llm_load_in_4bit,  # 4bit 양자화 로드 여부를 설정합니다.
    #     # token = "hf_...", # 게이트된 모델을 사용하는 경우 토큰을 사용하세요. 예: meta-llama/Llama-2-7b-hf
    # )
    # bnb_config = BitsAndBytesConfig(
    #     # load_in_4bit=True,
    #     # bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    model = GemmaForCausalLM.from_pretrained(
        args.llm_base_model,
        # quantization_config=bnb_config,
        device_map='auto',
        cache_dir=args.llm_cache_dir,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    train_loader, val_loader, test_loader, tokenizer, test_retrieval = dataloader_factory(args, dataset)
    model.config.use_cache = False
    trainer = LLMTrainer(args, model, train_loader, val_loader, test_loader, tokenizer, export_root, args.use_wandb)
    
    trainer.train()
    trainer.test(test_retrieval)


if __name__ == "__main__":
    args.model_code = 'llm'
    database=set_template(args)
    main(args, database, export_root=None)