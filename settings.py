from typing import Any, Callable, Set

from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict

class Settings(BaseSettings):

    model_config = ConfigDict(protected_namespaces=())

    ################################################################################
    # Base Model
    ################################################################################

    model_name: str = "models/TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    ################################################################################
    # QLoRA parameters
    ################################################################################

    # LoRA attention dimension
    lora_r: int = 16

    # Alpha parameter for LoRA scaling
    lora_alpha: int = 16

    # Dropout probability for LoRA layers
    lora_dropout: float = 0.1

    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    use_4bit: bool = True
    use_8bit: bool = False
    

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype: str = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type: str = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant: bool = False

    llm_int8_enable_fp32_cpu_offload: bool = True

    ################################################################################
    # TrainingArguments parameters
    ################################################################################

    output_dir: str = f"output/{model_name}"

    # Number of training epochs
    num_train_epochs: int = 1

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16: bool = False
    bf16: bool = True

    # Batch size per GPU for training
    per_device_train_batch_size: int = 1

    # Batch size per GPU for evaluation
    per_device_eval_batch_size: int = 1

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps: int = 1

    # Enable gradient checkpointing
    gradient_checkpointing: bool = True

    # Maximum gradient normal (gradient clipping)
    max_grad_norm: float = 0.3

    # Initial learning rate (AdamW optimizer)
    learning_rate: float = 2e-4

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay: float = 0.001

    # Optimizer to use
    optim: str = "paged_adamw_32bit"

    # Learning rate schedule
    lr_scheduler_type: str = "cosine"

    # Number of training steps (overrides num_train_epochs)
    max_steps: int = -1

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio: float = 0.03

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length: bool = True

    # Save checkpoint every X updates steps
    save_steps: int = 0

    # Log every X updates steps
    logging_steps: int = 25


    ################################################################################
    # SFT parameters
    ################################################################################

    # Maximum sequence length to use
    max_seq_length: int | None = None

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing: bool = False

    # Load the entire model on the GPU 0
    device_map: dict = {"": 0}

    ################################################################################
    # DATASET
    ################################################################################
    dataset_path: str = 'data/pe2_paragraph'
    
    
    ################################################################################
    # unsloth
    ################################################################################
    dtype: str | None = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit: bool | None = True # Use 4bit quantization to reduce memory usage. Can be False.
    load_pretrained: bool | None = True
