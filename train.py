import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_config, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    logging,
    pipeline,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

import chat_templates
from settings import Settings

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
    num_train_epochs=5,
    max_seq_length=1024,
    save_steps = 100
)

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(settings.model_name, trust_remote_code=True, add_eos_token=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

if 'llama-2' in settings.model_name.lower() or 'mistral' in settings.model_name.lower() or 'zephyr' in settings.model_name.lower():
    tokenizer.chat_template = chat_templates.llama_2
elif 'phi' in settings.model_name.lower():
    tokenizer.chat_template = chat_templates.phi2

dataset = load_dataset("text", data_dir=settings.dataset_path, sample_by="document", split="train")
dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(json.loads(x['text']), tokenize=False, add_generation_prompt=False)},  load_from_cache_file = False)


compute_dtype = getattr(torch, settings.bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=settings.use_4bit,
    load_in_8bit=settings.use_8bit,
    bnb_4bit_quant_type=settings.bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=settings.use_nested_quant,
    llm_int8_enable_fp32_cpu_offload=settings.llm_int8_enable_fp32_cpu_offload, ##new
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    # llm_int8_has_fp16_weight=True
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and settings.use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    settings.model_name,
    # quantization_config=bnb_config,
    device_map=settings.device_map,
    trust_remote_code=True,
    # flash_attn=True,
    # flash_rotary=True,
    # fused_dense=True, #for phi-2
)
model.config.use_cache = False
model.config.pretraining_tp = 1


if 'tiny' in settings.model_name.lower():
    response_template = "<|assistant|>"
elif 'llama-2' in settings.model_name.lower() or 'mistral' in settings.model_name.lower() or 'zephyr' in settings.model_name.lower():
    response_template = "[/INST]"
elif 'phi' in settings.model_name.lower():
    response_template = "Output:"


if 'tiny' in settings.model_name.lower():
    response_template_with_context = f"\n{response_template}"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
else:
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=settings.lora_alpha,
    lora_dropout=settings.lora_dropout,
    r=settings.lora_r,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    # target_modules= ["Wqkv", "out_proj"] #phi1.5, llama
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj', 'down_proj'] #tinyllama
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=settings.output_dir,
    num_train_epochs=settings.num_train_epochs,
    per_device_train_batch_size=settings.per_device_train_batch_size,
    gradient_accumulation_steps=settings.gradient_accumulation_steps,
    optim=settings.optim,
    save_steps=settings.save_steps,
    logging_steps=settings.logging_steps,
    learning_rate=settings.learning_rate,
    weight_decay=settings.weight_decay,
    fp16=settings.fp16,
    bf16=settings.bf16,
    max_grad_norm=settings.max_grad_norm,
    max_steps=settings.max_steps,
    warmup_ratio=settings.warmup_ratio,
    group_by_length=settings.group_by_length,
    lr_scheduler_type=settings.lr_scheduler_type,
    report_to="tensorboard",
    save_total_limit=3,
    # load_best_model_at_end = True
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="formatted_chat",
    max_seq_length=settings.max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=settings.packing,
    data_collator=collator,
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(settings.output_dir)