import argparse
import json
import os
import re
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
parser.add_argument('--lora_r', type=int, help='lora rank', default=16)
parser.add_argument('--lora_modules', type=str, help='lora rank', default='q_proj,k_proj,v_proj,gate_proj,up_proj,down_proj')
parser.add_argument('--bit4', action='store_true', help='user 4bit quantization', default=False)
parser.add_argument('--bit8', action='store_true', help='user 8bit quantization', default=False)
parser.add_argument('--load_pretrained', action='store_true', help='load from pretrained model', default=False)
parser.add_argument('--all_snapshots', action='store_true', help='we inference on all the snapshots in the given directory or not')

args = parser.parse_args()
use_lora = args.lora
output_extra_detail = ''
output_extra_detail += f"lora-r{args.lora_r}-{''.join([r[0] for r in args.lora_modules.split(',')])}" if use_lora else ""
output_extra_detail += f"-bs{args.batch_size}"
output_extra_detail += f"-ac{args.gradient_steps}"
output_extra_detail += f"-e{args.epochs}"
output_extra_detail += "-q4" if args.bit4 else ""
output_extra_detail += "-q8" if args.bit8 else ""
output_extra_detail += "-fp" if (not (args.bit4 and args.bit8)) else ""


settings = Settings(
    dataset_path = f'datasets/{args.dataset}',
    per_device_train_batch_size = args.batch_size,
    model_name = args.model_name,
    output_dir = f'output/{args.model_name.replace("/", "-")}-{output_extra_detail}-{args.dataset}',
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
    load_in_4bit = False,
    lora_r = args.lora_r
)

dirs = []
if args.all_snapshots:
    dirs = [os.path.join(settings.output_dir, d) for d in os.listdir(settings.output_dir) if os.path.isdir(os.path.join(settings.output_dir, d))]
    dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
else:
    dirs = [settings.output_dir]


for i, model_dir in enumerate(dirs):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_dir,
        max_seq_length = settings.max_seq_length,
        dtype = settings.dtype,
        load_in_4bit = settings.load_in_4bit,
    )

    FastLanguageModel.for_inference(model)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    if 'tiny' in settings.model_name.lower():
        tokenizer.chat_template = chat_templates.tiny_llama
    elif 'llama-2' in settings.model_name.lower() or 'mistral' in settings.model_name.lower() or 'zephyr' in settings.model_name.lower():
        tokenizer.chat_template = chat_templates.llama_2
    elif 'phi' in settings.model_name.lower():
        tokenizer.chat_template = chat_templates.phi2


    pred_dir = settings.output_dir.replace('output', 'preds')
    if args.all_snapshots:
        pred_dir = re.sub('-e\d+', f'-e{i+1:2d}')
    os.makedirs(pred_dir, exist_ok=True)

    counter = 0
    for root, _, files in os.walk(settings.dataset_path+"/test"):
        for f in tqdm(files):
            try:
                with open(os.path.join(root, f)) as f1, torch.no_grad():
                    sample = json.load(f1)
                    prompt = []
                    response_length = 0
                    for item in sample:
                        if item['role'] != 'assistant':
                            prompt.append(item)
                        else:
                            response_length = int(len(tokenizer(item['content'])['input_ids'])*1.3)

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
