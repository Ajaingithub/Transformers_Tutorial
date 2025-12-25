# FineTuning Llama for Chatting
# Learning how we can do Fine Tuning using this video from Krish Naik https://www.youtube.com/watch?v=iOdFUJiB0Zc
#region Loading Packages
# conda activate torch_gpu_dna
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, # To load the Causal lang. Model like ChatGPT
    AutoTokenizer, # To load the tokenization
    BitsAndBytesConfig, # for low memory convert to low precision like FP32 --> FP16
    HfArgumentParser,
    TrainingArguments, # passing Training arguments
    pipeline,
    logging,
)
from transformers import LlamaTokenizer
from peft import LoraConfig, PeftModel # Parameter efficient Fine Tuning. LORA performs matrix decomposition which makes the matrix quite small. There is QLORA which works on Quantization LORA.
from trl import SFTTrainer # Train transformer language models with reinforcement learning.


#region Defining Arguments
# The model that you want to train from the Hugging Face hub
# Non-chat version → worse conversational behavior unless heavily fine-tuned.
model_name = "NousResearch/Llama-2-7b-chat-hf"


# The instruction dataset to use
# Small (~1k examples) → good for quick adaptation, not general knowledge learning.
dataset_name = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model name
new_model = "Llama-2-7b-chat-finetune"

################################################################################
# QLoRA parameters
################################################################################

# LoRA attention dimension
lora_r = 8 ## This is the rank by which matrix decomposition takes place. 
# suppose if lora_r = 1 then 3 X 3 matrix is decomposed to 3 X 1 and 1 X 3.

# Alpha parameter for LoRA scaling
# Scaling factor.
# Effective LoRA contribution ≈ alpha / r = 16 / 64 = 0.25
lora_alpha = 16 # scaling factor

# Dropout probability for LoRA layers
lora_dropout = 0.1

################################################################################
# bitsandbytes parameters
################################################################################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Even though your model weights are stored in 4-bit, the actual matrix multiplications are NOT done in 4-bit.
# Instead:
# Weights are dequantized on the fly 
# computation happens in float16
# 4-bit weights  →  dequantize →  float16 compute → output


# Quantization type (fp4 or nf4)
# NF4 (NormalFloat4) is optimized for neural network weights.
# Better accuracy than FP4. 
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

# Original fp16 model
#         ↓
# Quantized to NF4 (4-bit)   ← bnb_4bit_quant_type="nf4"
#         ↓
# Stored efficiently in GPU memory
#         ↓
# Dequantized to fp16        ← bnb_4bit_compute_dtype="float16"
#         ↓
# Matrix multiplication & gradients


################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False # (half precision) 1 bit sign, 5 bits exponent, 10 bits mantissa (fraction)
bf16 = False # 1 bit sign, 8 bits exponent, 7 bits mantissa
# Same exponent range as fp32
# Much more numerically stable than fp16
# Lower precision than fp16
# Large-scale model training (TPUs, modern GPUs like A100, H100)

# Batch size per GPU for training
per_device_train_batch_size = 1

# Batch size per GPU for evaluation
per_device_eval_batch_size = 1

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 4
# This keeps effective batch size = 4 without OOM. 
# (forward → backward) × N → optimizer.step() where N is the number of minibatches 

# Enable gradient checkpointing
gradient_checkpointing = True
# Training transformers normally stores all intermediate activations so gradients can be computed during backprop.
# For a 7B model, this eats huge GPU memory.
# Gradient checkpointing:
# Does NOT store all activations
# Recomputes them during backward pass instead
# Trades compute time for memory savings
# so instead of Forward → store activations → Backward
# It do
# Forward → discard activations
# Backward → recompute forward → compute gradients


# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001
# Weight decay is a regularization technique that prevents a model from overfitting by discouraging weights from becoming too large.
# Think of it as telling the model:
# “Fit the data well, but don’t rely on extremely large parameter values.”
# During training, models can reduce loss by:
# Making some weights very large
# Memorizing training examples
# Weight decay penalizes large weights, pushing them slightly toward zero at every update.
# Without weight decay:
# w = w - lr * gradient
# With weight decay:
# w = w - lr * (gradient + λ * w)

# Optimizer to use
optim = "paged_adamw_32bit"

# Learning rate schedule
lr_scheduler_type = "cosine"
# Early training → need large steps to learn quickly
# Later training → need small steps to fine-tune
# Too large LR → divergence
# Too small LR → slow or stuck training

# Number of training steps (overrides num_train_epochs)
max_steps = -1


# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

# Log every X updates steps
logging_steps = 25

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
max_seq_length = 1024

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"": 0}

#region Configuring
# Load dataset 
dataset = load_dataset(dataset_name, split="train") ## Dataset to load

# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype) # bnb_4bit_compute_dtype = float 16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit, # Quantization is performed at 4 bits
    bnb_4bit_quant_type=bnb_4bit_quant_type, # nf4 that is more precised
    bnb_4bit_compute_dtype=compute_dtype, # float 16 caluclation is perform with fp16 for better accuracy
    bnb_4bit_use_double_quant=use_nested_quant, # use_nested_quant = False
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    print(major)
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)
    else:
        print("Your GPU does not supports bfloat16: accelerate training with bf16=True")
# CUDA compute capability:
# 7.x → Volta / Turing (V100, T4)
# 8.0+ → Ampere (A100)
# 9.0+ → Hopper (H100)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,  # Load the entire model on the GPU 0
    trust_remote_code=True
)

print(model)
print(model.config)


model.config.use_cache = False
model.config.pretraining_tp = 1

# since we saved using old version, we just need to upload it on the newer version.
from transformers import AutoTokenizer
os.chdir("/mnt/data/projects/.immune/Personal/Transformers_Tutorial/Llama/")

tokenizer = AutoTokenizer.from_pretrained(
    "llama2_tokenizer_frozen",
    use_fast=True
)
print(tokenizer)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)


# Set training parameters
# https://huggingface.co/docs/transformers/v5.0.0rc1/en/main_classes/trainer#transformers.TrainingArguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

print("Training Arguments\n")
print(training_arguments,"\n")
print("Datasets\n")
print(dataset['text'][0:10])

# Set supervised fine-tuning (SFT) parameters
# https://huggingface.co/docs/trl/en/sft_trainer#expected-dataset-type-and-format
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_arguments,
)

print("Training the Model")
trainer.train()