import argparse
import json
import os
from math import floor

import torch
from tqdm import tqdm
from unsloth import FastLanguageModel

import chat_templates
from settings import Settings

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, help='The dataset root folder', default='aae2/adus')
parser.add_argument('--model_name', type=str, help='LLM Model name or path', default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
parser.add_argument('--epochs', type=int, help='number of epochs', default=1)
parser.add_argument('--batch_size', type=int, help='train batch size', default=8)
parser.add_argument('--gradient_steps', type=int, help='gradient_accumulation_steps', default=1)
parser.add_argument('--lora', action='store_true', help='user lora', default=True)
parser.add_argument('--start_percent', type=int, help='user lora', default=0)
parser.add_argument('--end_percent', type=int, help='user lora', default=25)

args = parser.parse_args()
use_lora = args.lora

settings = Settings(
    dataset_path = f'datasets/{args.dataset}',
    per_device_train_batch_size = args.batch_size,
    model_name = args.model_name,
    output_dir = f'output/{args.model_name.replace("/", "-")}{"-lora" if use_lora else ""}-{args.dataset}',
    use_4bit = False,
    use_8bit = False,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    gradient_accumulation_steps = args.gradient_steps,
    llm_int8_enable_fp32_cpu_offload = True,
    per_device_eval_batch_size = 4,
    num_train_epochs=args.epochs,
    max_seq_length=1024,
    save_steps = 1000,
    load_in_4bit = False
)


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = settings.output_dir,
    max_seq_length = settings.max_seq_length,
    dtype = settings.dtype,
    load_in_4bit = settings.load_in_4bit,
)


tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

if 'tiny' in settings.model_name.lower():
    tokenizer.chat_template = chat_templates.tiny_llama
elif 'llama-2' in settings.model_name.lower() or 'mistral' in settings.model_name.lower() or 'zephyr' in settings.model_name.lower():
    tokenizer.chat_template = chat_templates.llama_2
elif 'phi' in settings.model_name.lower():
    tokenizer.chat_template = chat_templates.phi2


pred_dir = settings.output_dir.replace('output', 'preds')
os.makedirs(pred_dir, exist_ok=True)

counter = 0
for root, _, files in os.walk(settings.dataset_path+"/test"):
    for f in tqdm(files[floor(len(files) * (args.start_percent/100)): floor(len(files) * (args.end_percent/100))]):
        try:
            with open(os.path.join(root, f)) as f1, torch.no_grad():
                sample = json.load(f1)
                prompt = []
                response_length = 0
                for item in sample:
                    if item['role'] != 'assistant':
                        prompt.append(item)
                    else:
                        response_length = int(len(item['content'].split())*1.5)
                        
                inputs = tokenizer.apply_chat_template(prompt, return_tensors='pt', tokenize=True, add_generation_prompt=True).to('cuda')
                output = model.generate(input_ids = inputs, max_new_tokens=response_length)
                response = tokenizer.decode(output[0].tolist())
            # result = generate(prompt)
            # print(response)
            with open(f"{pred_dir}/{f.replace('.json', '')}.txt", 'w') as f2:
                f2.write(response)
        except Exception as e:
            print(e)
            print(prompt)
            print(f)
