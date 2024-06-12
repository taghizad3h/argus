import json
import os

import torch
from peft import LoraConfig, PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    logging,
)

import chat_templates
from settings import Settings

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

dataset = 'aae2/adus'
model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
use_lora = True

settings = Settings(
    dataset_path = f'datasets/{dataset}',
    per_device_train_batch_size = 1,
    # model_name = 'models/microsoft/phi-2',
    model_name = model_name,
    # output_dir = 'output/phi-28bitqlora',
    output_dir = f'output/{model_name.replace("/", "-")}{"-lora" if use_lora else ""}-{dataset}',
    use_4bit = False,
    use_8bit = False,
    gradient_accumulation_steps = 4,
    llm_int8_enable_fp32_cpu_offload = True,
    per_device_eval_batch_size = 4,
    num_train_epochs=10,
    max_seq_length=1024,
    save_steps = 100
)

print(settings.dataset_path)

compute_dtype = getattr(torch, settings.bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=settings.use_4bit,
    load_in_8bit=settings.use_8bit,
    bnb_4bit_quant_type=settings.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=settings.use_nested_quant,
)


# Load the model (use bf16 for faster inference)
model = AutoModelForCausalLM.from_pretrained(
    settings.model_name,
    # quantization_config=bnb_config,
    device_map=settings.device_map,
    trust_remote_code=True,
    # flash_attn=True, 
    # flash_rotary=True, 
    # fused_dense=True #for phi-2
)

tokenizer = AutoTokenizer.from_pretrained(settings.model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training


if 'llama-2' in settings.model_name.lower():
    tokenizer.chat_template = chat_templates.llama_2
elif 'phi' in settings.model_name.lower():
    tokenizer.chat_template = chat_templates.phi2


# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=settings.lora_alpha,
    lora_dropout=settings.lora_dropout,
    r=settings.lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    # target_modules= ["Wqkv", "out_proj"] #phi1.5, llama
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj', 'down_proj'] #tinyllama
)

model = PeftModel.from_pretrained(model, model_id = settings.output_dir, config = peft_config)
# model = model.merge_and_unload()
model.to('cuda')
model.eval()

# pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100, do_sample=False)

# test_dataset = load_dataset('text', data_dir=settings.dataset_path, sample_by="document", split='test')

# def generate(user_question):
#     result = pipe(user_question, return_full_text=False)
#     return result[0]['generated_text']


pred_dir = settings.output_dir.replace('output', 'preds')
os.makedirs(pred_dir, exist_ok=True)

for root, _, files in os.walk(settings.dataset_path+"/test"):
    for f in tqdm(files):
        try:
            with open(os.path.join(root, f)) as f1, torch.no_grad():
                sample = json.load(f1)
                prompt = []
                for item in sample:
                    if item['role'] != 'assistant':
                        prompt.append(item)
                inputs = tokenizer.apply_chat_template(prompt, return_tensors='pt', tokenize=True, add_generation_prompt=True).to('cuda')
                output = model.generate(input_ids = inputs, max_new_tokens=5)
                response = tokenizer.decode(output[0].tolist())
            # result = generate(prompt)
            # print(response)
            with open(f"{pred_dir}/{f.replace('.json', '')}.txt", 'w') as f2:
                f2.write(response)
        except Exception as e:
            print(e)
            print(prompt)
            print(f)
