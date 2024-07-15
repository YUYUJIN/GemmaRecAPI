from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
import os
import shutil
from distutils.dir_util import copy_tree
import random

from config import *
from modelStructure import *
from dataloader import *
from dataBaseSet.survey import SurveyDataset
from trainer import *
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

from modelStructure import GemmaForCausalLM
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
os.environ["WANDB_MODE"] = "dryrun"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from pytorch_lightning import seed_everything

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

# database
@app.route('/insert',methods=['POST'])
def insert():
    if request.method=='POST':
        data=request.get_json()
        table=data['table']
        if table=='survey':
            database.insert_data('survey',['uid','sid','rating','comment'],data['data'])
        elif table=='item':
            database.insert_data('items',['sid','item','style'],data['data'],'sid')
        elif table=='userinfo':
            database.insert_data('userinfo',['uid','style','sex','job','married'],data['data'],'uid')

        response={'message':'Now database is updated, please call /train API to update rec model.'}
        return jsonify(response),200
    else:
        return None

@app.route('/train',methods=['POST'])
def train_all():
    if request.method=='POST':
        data=request.get_json()
        # set train parameters
        args.num_epochs=data.get('num_epochs') if data.get('num_epochs') else args.num_epochs
        args.optimizer=data.get('optimizer') if data.get('optimizer') else args.optimizer
        args.weight_decay=data.get('weight_decay') if data.get('weight_decay') else args.weight_decay
        args.lr=data.get('lr') if data.get('lr') else args.lr
        args.max_grad_norm=data.get('max_grad_norm') if data.get('max_grad_norm') else args.max_grad_norm
        args.enable_lr_schedule=data.get('enable_lr_schedule') if data.get('enable_lr_schedule') else args.enable_lr_schedule
        args.decay_step=data.get('decay_step') if data.get('decay_step') else args.decay_step
        args.early_stopping=data.get('early_stopping') if data.get('early_stopping') else args.early_stopping
        args.early_stopping_patience=data.get('early_stopping_patience') if data.get('early_stopping_patience') else args.early_stopping_patience

        # set lru parameters
        args.bert_max_len=data.get('bert_max_len') if data.get('bert_max_len') else args.bert_max_len
        args.bert_hidden_units=data.get('bert_hidden_units') if data.get('bert_hidden_units') else args.bert_hidden_units
        args.bert_num_blocks=data.get('bert_num_blocks') if data.get('bert_num_blocks') else args.bert_num_blocks
        args.bert_num_heads=data.get('bert_num_heads') if data.get('bert_num_heads') else args.bert_num_heads
        args.bert_head_size=data.get('bert_head_size') if data.get('bert_head_size') else args.bert_head_size
        args.bert_dropout=data.get('bert_dropout') if data.get('bert_dropout') else args.bert_dropout
        args.bert_attn_dropout=data.get('bert_attn_dropout') if data.get('bert_attn_dropout') else args.bert_attn_dropout
        args.bert_mask_prob=data.get('bert_mask_prob') if data.get('bert_mask_prob') else args.bert_mask_prob

        seed_everything(args.seed)
        dataset=SurveyDataset(args,database).load_dataset()

        # train lru
        args.model_code='lru'
        train_loader, val_loader, test_loader = dataloader_factory(args,dataset)
        model = LRURec(args)
        export_root = EXPERIMENT_ROOT + '/' + args.dataset_code
        
        args.train_batch_size = 64
        args.val_batch_size = 64
        args.test_batch_size = 64
        trainer = LRUTrainer(args, model, train_loader, val_loader, test_loader, export_root, args.use_wandb)
        trainer.train()
        lru_test_score=trainer.test()
        #the next line generates val / test candidates for reranking
        trainer.generate_candidates(os.path.join(export_root, 'retrieved.pkl'))

        # train llm
        if data['train_type']=='all':
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            model = GemmaForCausalLM.from_pretrained(
                args.llm_base_model,
                quantization_config=bnb_config,
                device_map='auto',
                cache_dir=args.llm_cache_dir,
                token=args.hf_token
            )

            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': True})
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
            
            args.model_code='llm'
            args.train_batch_size = 16
            args.val_batch_size = 4
            args.test_batch_size = 4
            train_loader, val_loader, test_loader, tokenizer, test_retrieval = dataloader_factory(args, dataset, export_root)
            model.config.use_cache = False
            trainer = LLMTrainer(args, model, train_loader, val_loader, test_loader, tokenizer, export_root, args.use_wandb)

            trainer.train()
            llm_test_score=trainer.test(test_retrieval)
            
        else:
            llm_test_score=None
        
        # save maps
        with open(os.path.join(args.save_folder_path,'maps.json'),'w') as f:
            json.dump({'umap':dataset['umap'], 'smap': dataset['smap']}, f)
        with open(os.path.join(args.save_folder_path,'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        
        # save best model
        shutil.copyfile(os.path.join(export_root,'models','best_acc_model.pth'), os.path.join(args.save_folder_path,'best_acc_model.pth'))
        if data['train_type']=='all':
            files=os.listdir(export_root)
            epoch=0
            for file in files:
                check=file.split('-')
                if check[0]=='checkpoint' and int(check[1])>epoch:
                    epoch=int(check[1])
            copy_tree(os.path.join(export_root,f'checkpoint-{epoch}'), os.path.join(args.save_folder_path,'llm'))

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        response={'lru_test_score':lru_test_score,
                  'llm_test_score':llm_test_score}
        return jsonify(response),200
    else:
        return None

@app.route('/recommend',methods=['POST'])
def inference():
    if request.method=='POST':
        data=request.get_json()
        # set environ to recently training environ
        args_inf=argparse.ArgumentParser()
        with open(os.path.join(args.save_folder_path,'args.json'),'r') as f:
            args_inf.__dict__ = json.load(f)
        with open(os.path.join(args.save_folder_path,'maps.json'),'r') as f:
            maps=json.load(f)
        uid=data['uid']
        rows=database.get_data('surveygroup','sid',f'uid={uid}')
        sids=[]
        for row in rows:
            if maps['smap'].get(str(row[0])):
                sids.append(maps['smap'][str(row[0])]) 
        seq = sids[-args_inf.bert_max_len:]
        padding_len = args_inf.bert_max_len - len(seq)
        seq = [0] * padding_len + seq
        seq=torch.LongTensor([seq]).to(args_inf.device)
        
        # lru inference
        args_inf.model_code='lru'
        lru_model= LRURec(args_inf)
        best_model_dict = torch.load(os.path.join(
            args.save_folder_path, 'best_acc_model.pth')).get(STATE_DICT_KEY)
        lru_model.load_state_dict(best_model_dict)
        lru_model.to(args_inf.device)
        lru_model.eval()
        with torch.no_grad():
            scores = lru_model(seq)[0, -1, :]
            _, L = seq.shape
            for i in range(L):
                scores[seq[0, i]] = -1e9
            scores[0] = -1e9  # padding
            candidates=torch.topk(torch.tensor(scores), args_inf.llm_negative_sample_size+1).indices.tolist()
            
        # llm inference
        # prepare recent llm model
        args_inf.model_code='llm'
        inv_smap={v: k for k, v in maps['smap'].items()}
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        model = GemmaForCausalLM.from_pretrained(
            args_inf.llm_base_model,
            quantization_config=bnb_config,
            device_map='auto',
            cache_dir=args_inf.llm_cache_dir,
            token=args_inf.hf_token
        )
        model=PeftModel.from_pretrained(model, os.path.join(args_inf.save_folder_path,'llm'), device_map='auto', torch_dtype=torch.bfloat16)
        model.merge_and_unload()
        tokenizer=AutoTokenizer.from_pretrained(os.path.join(args_inf.save_folder_path,'llm'), cache_dir=args_inf.llm_cache_dir)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
        tokenizer.clean_up_tokenization_spaces = True
        verbalizer = ManualVerbalizer(
            tokenizer=tokenizer,
            prefix='',
            post_log_softmax=False,
            classes=list(range(args_inf.llm_negative_sample_size+1)),
            label_words={i: chr(ord('A')+i) for i in range(args_inf.llm_negative_sample_size+1)},
        ).to(args_inf.device)
        model.eval()

        # prepare prompt from database
        def truncate_text(text):
            text_ = tokenizer.tokenize(text)[:args_inf.llm_max_title_len]
            text = tokenizer.convert_tokens_to_string(text_)
            return text
        row=database.get_data('usergroup',option=f'uid={uid}')[0]
        info=truncate_text(f'{row[2]}, {row[4]}, {row[3]}, {row[1]}')
        sids=tuple(int(inv_smap[sid]) for sid in sids)
        _comments=database.get_data('surveygroup','comment',f'uid={uid} and sid in {sids}')
        if len(_comments)>args_inf.llm_max_history:
            comments=random.sample(_comments,args_inf.llm_max_history)
        else:
            comments=_comments
        seq=' \n '.join(['(' + str(idx + 1) + ') ' + truncate_text(comment[0]) for idx,comment in enumerate(comments)])
        sids=tuple(int(inv_smap[sid]) for sid in candidates)
        items=database.get_data('items','item',f'sid in {sids}')
        can=' \n '.join(['(' + chr(ord('A') + idx) + ') ' + truncate_text(item[0]) for idx, item in enumerate(items)])
        prompt=Prompter().generate_prompt(instruction=args_inf.llm_system_template,input=args_inf.llm_input_template.format(info, seq, can))
        tokenized_full_prompt=tokenizer(prompt, truncation=True, max_length=args_inf.llm_max_text_len,padding=False,return_tensors='pt')

        # data to model input
        input_ids = tokenized_full_prompt['input_ids']
        attention_mask = tokenized_full_prompt['attention_mask']
        if len(input_ids) > args_inf.llm_max_text_len:
            input_ids = input_ids[-args_inf.llm_max_text_len:]
            attention_mask = attention_mask[-args_inf.llm_max_text_len:]
        elif len(input_ids) <= args_inf.llm_max_text_len:
            padding_length = args_inf.llm_max_text_len - len(input_ids)
            input_ids = torch.cat([torch.zeros((input_ids.shape[0], padding_length), dtype=torch.long), input_ids], dim=1)
            attention_mask = torch.cat([torch.zeros((attention_mask.shape[0], padding_length), dtype=torch.long), attention_mask], dim=1)
        input={'input_ids':input_ids.to(args_inf.device),'attention_mask':attention_mask.to(args_inf.device)}
        
        with torch.no_grad():
            result=model(**input)
            ranks=(-verbalizer.process_logits(result.logits)).argsort(dim=1)[0]
        
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        response={'item_titles':[items[rank] for rank in ranks[:data['k']]],
            'error_message':None}
        return jsonify(response),200
    else:
        return None


if __name__ == '__main__':
    load_dotenv()
    args.host,args.user,args.password,args.dbname,args.port=os.environ.get('HOST'),os.environ.get('USERNAME'),os.environ.get('PASSWARD'),os.environ.get('DATABASENAME'),int(os.environ.get('PORT'))
    args.hf_token=os.environ.get('HFTOKEN')
    database=set_template(args)
    app.run(host='0.0.0.0',port=5000,debug=False)