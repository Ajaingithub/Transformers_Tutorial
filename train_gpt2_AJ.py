# conda activate torch_gpu
import os
import pandas as pd
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        # Trying to make it more like GPT2 so initialization has set the distribution as 0.2 std dev
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(torch.arange(T, dtype=torch.long, device = idx.device))
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            # print(k)
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        wd_param = [p for n, p in param_dict.items() if p.dim() >= 2]
        nwd_param = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': wd_param, 'weight_decay': weight_decay},
            {'params': nwd_param, 'weight_decay': 0},
        ]
        # weight decay is a regularization, by distributing the weight across the channels
        num_decay_params = sum(p.numel() for p in wd_param)
        num_ndecay_params = sum(p.numel() for p in nwd_param)
        print(f'number of deacy parameters = {len(wd_param)} with {num_ndecay_params}')
        print(f'number of non-deacy parameters= {len(nwd_param)} with {num_ndecay_params}')
        # Kernel fusion of the AdamW update
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-08, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK',-1)) != -1 # is this a ddp run?
if ddp:
    assert torch.cuda.is_available() # ddp can run only on CUDA
    init_process_group(backend = 'nccl') # initiation
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    # vanilla non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 0
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

## Now for training you surely need to have GPU or cuda
# device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends,"mps") and torch.backends.mps.is_available():
    device = "mps"

print("device name",device)

# import sys; sys.exit(0)

## This will pass all the weights and bias from gpt2 to our model
model = GPT(GPTConfig(vocab_size=50304)) # since 50304 is the in the power of 2
model.to(device)

import tiktoken

class DataloaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        os.chdir("/mnt/data/projects/.immune/Personal/Transformers_Tutorial/")
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded tokens {len(self.tokens)} tokens')
        print(f'1 epoch has these token {len(self.tokens) // (B * T)} batches')
        
        # State
        self.current_position = self.B * self.T * self.process_rank
 
    def next_batch(self):
        B=self.B
        T=self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = buf[:-1].view(B,T) ### These are tokens to be used to predict the new token key
        y = buf[1:].view(B,T) ## This is the target token query
        # advance to the new position when it is completed
        self.current_position += B * T * self.num_processes
        # if loading the next batch is greater than the token present 
        if (self.current_position + (self.B * self.T * self.process_rank + 1) > len(self.tokens)):
            self.current_position = self.B * self.T * self.process_rank
        return x,y

### Using the dataloader and updated the forward method in the model
model = GPT(GPTConfig(vocab_size = 50304))
model.to(device)
# model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids = [ddp_local_rank])

# torch.set_float32_matmul_precision('high') # Does not work on my TESLA T4 but will work on A100
# create a PyTorch optimizer
import torch
import time
import math
import inspect

total_batch_size = 524288
B = 4 # minor_batch_size
T = 1024 #token_size
assert total_batch_size % (B * T) == 0 # make sure remainder is 0
grad_accum_steps = total_batch_size / (B * T)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas = (0.9, 0.95), eps=1e-08)
optimizer = model.configure_optimizers(weight_decay = 0.1, learning_rate = 6e-04, device = device)
train_loader = DataloaderLite(B = B, T = T, process_rank=ddp_rank, num_processes = ddp_world_size)
loss_accum = 0.0
for step in range(max_steps): 
    torch.cuda.empty_cache()
    t0 = time.time()
    optimizer.zero_grad(set_to_none = True)
    # with torch.autocast(device_type=device, dtype=torch.bfloat16): # require more memory which I donot have
    for microstep in range(int(grad_accum_steps)):
        x,y = train_loader.next_batch()
        x,y = x.to(device), y.to(device)
        logits, loss = model(x,y)
        loss = loss / float(grad_accum_steps) ## Act as a normalizer
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() ### It lets the GPU finish the job
    t1 = time.time()
    dt = (t1 - t0) * 1000 # milliseconds
    token_ps = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step = {step:4d}, loss = {loss.item():.6f},  norm {norm:.4f}, lr {lr:.4e}, time take {dt:.2f}ms, tok/sec {token_ps}")


# /mnt/data/tools/miniconda3/envs/torch_gpu/lib/python3.10/site-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
#   warnings.warn(
# using device: cuda
# device name cuda
# number of deacy parameters = 50 with 121344
# number of non-deacy parameters= 98 with 121344
# loaded tokens 338025 tokens
# 1 epoch has these token 82 batches
# step =    0, loss = 0.085162,  norm 33.3399, lr 6.0000e-05, time take 123545.55ms, tok/sec 33.15376499434342
# step =    1, loss = 0.073866,  norm 9.3363, lr 1.2000e-04, time take 131423.91ms, tok/sec 31.166321897943607
# step =    2, loss = 0.070087,  norm 3.5702, lr 1.8000e-04, time take 132495.40ms, tok/sec 30.914279974458132
# step =    3, loss = 0.071473,  norm 7.6058, lr 2.4000e-04, time take 133326.63ms, tok/sec 30.72154526815539
# step =    4, loss = 0.066641,  norm 2.8598, lr 3.0000e-04, time take 133337.52ms, tok/sec 30.719034295495263
# step =    5, loss = 0.065473,  norm 5.5698, lr 3.6000e-04, time take 132449.71ms, tok/sec 30.92494432401095
# step =    6, loss = 0.062012,  norm 2.4300, lr 4.2000e-04, time take 132686.46ms, tok/sec 30.8697671530178
# step =    7, loss = 0.059120,  norm 2.7769, lr 4.8000e-04, time take 132665.99ms, tok/sec 30.874530362558847
# step =    8, loss = 0.055264,  norm 1.5479, lr 5.4000e-04, time take 133763.27ms, tok/sec 30.621261110201775
# step =    9, loss = 0.109916,  norm 57.6696, lr 6.0000e-04, time take 133684.70ms, tok/sec 30.639257892832806
# step =   10, loss = 0.050185,  norm 4.3599, lr 6.0000e-04, time take 133788.60ms, tok/sec 30.615462961628072
# step =   11, loss = 0.048459,  norm 2.8958, lr 5.9917e-04, time take 135138.58ms, tok/sec 30.309626846909474
# step =   12, loss = 0.045981,  norm 2.3397, lr 5.9668e-04, time take 134801.28ms, tok/sec 30.385468897358155
# step =   13, loss = 0.045287,  norm 3.4850, lr 5.9254e-04, time take 133018.02ms, tok/sec 30.792820917820748
# step =   14, loss = 0.043771,  norm 2.2819, lr 5.8679e-04, time take 133649.88ms, tok/sec 30.647241386190064
# step =   15, loss = 0.042485,  norm 1.4959, lr 5.7945e-04, time take 133321.18ms, tok/sec 30.722800688660218
# step =   16, loss = 0.041980,  norm 2.1488, lr 5.7057e-04, time take 133064.12ms, tok/sec 30.782152837741915
# step =   17, loss = 0.040923,  norm 0.8294, lr 5.6021e-04, time take 133693.34ms, tok/sec 30.637277585023707
# step =   18, loss = 0.042364,  norm 7.9678, lr 5.4843e-04, time take 133485.31ms, tok/sec 30.685024753001354
# step =   19, loss = 0.040352,  norm 3.4934, lr 5.3531e-04, time take 134128.03ms, tok/sec 30.537986551153182
# step =   20, loss = 0.039683,  norm 1.2628, lr 5.2092e-04, time take 133737.50ms, tok/sec 30.627160718073263
# step =   21, loss = 0.039385,  norm 2.1853, lr 5.0535e-04, time take 134285.52ms, tok/sec 30.502171339699405
# step =   22, loss = 0.038585,  norm 1.2733, lr 4.8870e-04, time take 134631.28ms, tok/sec 30.423836945367125
# step =   23, loss = 0.038617,  norm 2.8311, lr 4.7107e-04, time take 134080.10ms, tok/sec 30.54890218456703
# step =   24, loss = 0.038139,  norm 2.4171, lr 4.5258e-04, time take 134436.23ms, tok/sec 30.467977823690426
# step =   25, loss = 0.037588,  norm 1.4182, lr 4.3332e-04, time take 134786.66ms, tok/sec 30.388764270618413
# step =   26, loss = 0.037196,  norm 1.4951, lr 4.1343e-04, time take 134470.40ms, tok/sec 30.460235525476193
# step =   27, loss = 0.036763,  norm 1.2718, lr 3.9303e-04, time take 134633.13ms, tok/sec 30.423418376171504
# step =   28, loss = 0.036213,  norm 0.9692, lr 3.7224e-04, time take 134338.87ms, tok/sec 30.490058883744553
# step =   29, loss = 0.035897,  norm 1.7957, lr 3.5118e-04, time take 134721.51ms, tok/sec 30.40345966114796
# step =   30, loss = 0.035258,  norm 0.8130, lr 3.3000e-04, time take 134275.38ms, tok/sec 30.504474525565517
# step =   31, loss = 0.035161,  norm 1.9839, lr 3.0882e-04, time take 134653.23ms, tok/sec 30.418877621841265
# step =   32, loss = 0.034712,  norm 1.5581, lr 2.8776e-04, time take 134664.13ms, tok/sec 30.41641351006247
# step =   33, loss = 0.034203,  norm 1.3260, lr 2.6697e-04, time take 134369.52ms, tok/sec 30.4831032956689
# step =   34, loss = 0.033779,  norm 1.2169, lr 2.4657e-04, time take 134829.14ms, tok/sec 30.379190414404754
# step =   35, loss = 0.033459,  norm 1.7110, lr 2.2668e-04, time take 134484.75ms, tok/sec 30.456985108411764
# step =   36, loss = 0.033083,  norm 1.7888, lr 2.0742e-04, time take 134855.17ms, tok/sec 30.3733253697333
# step =   37, loss = 0.032581,  norm 0.6932, lr 1.8893e-04, time take 136517.58ms, tok/sec 30.003462027696067
# step =   38, loss = 0.032309,  norm 0.9966, lr 1.7130e-04, time take 135075.22ms, tok/sec 30.3238447194389
# step =   39, loss = 0.031924,  norm 0.9943, lr 1.5465e-04, time take 134908.91ms, tok/sec 30.361226581898798
# step =   40, loss = 0.031621,  norm 2.0147, lr 1.3908e-04, time take 135033.91ms, tok/sec 30.333121769674495
# step =   41, loss = 0.031332,  norm 1.6958, lr 1.2469e-04, time take 134759.43ms, tok/sec 30.39490415325679
# step =   42, loss = 0.031022,  norm 1.0203, lr 1.1157e-04, time take 134905.71ms, tok/sec 30.361946234023524
# step =   43, loss = 0.030696,  norm 1.3814, lr 9.9787e-05, time take 134899.31ms, tok/sec 30.363387840836793
# step =   44, loss = 0.030482,  norm 1.7922, lr 8.9428e-05, time take 134717.28ms, tok/sec 30.404415383351797
# step =   45, loss = 0.030235,  norm 1.0840, lr 8.0553e-05, time take 134884.90ms, tok/sec 30.366632158031788
# step =   46, loss = 0.029987,  norm 2.1959, lr 7.3215e-05, time take 134674.47ms, tok/sec 30.414079935520242
# step =   47, loss = 0.029703,  norm 1.1625, lr 6.7460e-05, time take 134817.25ms, tok/sec 30.38186840993378