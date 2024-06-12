import json

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from unsloth import FastLanguageModel

import chat_templates
from settings import Settings


dataset = 'aae2/adus'
model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
use_lora = True

settings = Settings(
    dataset_path = f'datasets/{dataset}',
    per_device_train_batch_size = 4,
    model_name = model_name,
    output_dir = f'output/{model_name.replace("/", "-")}{"-lora" if use_lora else ""}-{dataset}',
    use_4bit = False,
    use_8bit = False,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    gradient_accumulation_steps = 1,
    llm_int8_enable_fp32_cpu_offload = True,
    per_device_eval_batch_size = 1,
    num_train_epochs=5,
    max_seq_length=1024,
    save_steps = 1000,
    load_in_4bit = False
)


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = settings.model_name,
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


if 'tiny' in settings.model_name.lower():
    response_template = "<|assistant|>"
elif 'llama-2' in settings.model_name.lower() or 'mistral' in settings.model_name.lower() or 'zephyr' in settings.model_name.lower():
    response_template = "[/INST]"
elif 'phi' in settings.model_name.lower():
    response_template = "Output: "
elif 'llama-3' in settings.model_name.lower():
    response_template = '<|start_header_id|>assistant<|end_header_id|>\n\n'


model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_alpha = settings.lora_alpha,
    lora_dropout = settings.lora_dropout, # Supports any, but = 0 is optimized
#     bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
)

model.train()

print(model)


dataset = load_dataset("text", data_dir=settings.dataset_path, sample_by="document", split="train")
dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(json.loads(x['text']), tokenize=False, add_generation_prompt=False)}, load_from_cache_file = False)


if 'tiny' in settings.model_name.lower():
    response_template_with_context = f"\n{response_template}"
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
else:
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


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