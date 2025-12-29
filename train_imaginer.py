import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import math
import time

# --- CONFIGURATION ---
DATA_PATH = "train_data_final.pt" # Updated filename
BATCH_SIZE = 8 # Keep this low for Mac/MPS stability
BLOCK_SIZE = 512 # Context window size
LEARNING_RATE = 3e-4 # Learning rate
MAX_ITERS = 30000 # SCALED UP: 2.5k -> 30k (This is roughly 2.5 epochs)
VAL_INTERVAL = 1000 # Validate every 1000 iterations
SAVE_INTERVAL = 5000 # Save every 5000 iterations

if torch.cuda.is_available(): DEVICE = 'cuda'
elif torch.backends.mps.is_available(): DEVICE = 'mps'
else: DEVICE = 'cpu'
print(f"Using Device: {DEVICE}")

# --- 1. DATASET WITH METADATA LOADING ---
class ImaginationDataset(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path, weights_only=False)
        self.samples = data['samples']
        self.meta = data['meta'] # Load the Truth
        
        self.vocab_size = self.meta['vocab_size_total']
        self.img_start_id = self.meta['img_start_id']
        self.img_end_id = self.meta['img_end_id']
        
        print(f"Loaded Data. Vocab: {self.vocab_size}, START: {self.img_start_id}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        if len(seq) > BLOCK_SIZE:
            seq = seq[-BLOCK_SIZE:]

        x = seq[:-1].clone()
        y = seq[1:].clone()

        # Mask Prompt using the LOADED ID
        img_start_pos = (x == self.img_start_id).nonzero(as_tuple=True)[0]
        if len(img_start_pos) > 0:
            p = img_start_pos[0].item()
            y[:p] = -100 # IGNORE_INDEX
        
        pad_len = BLOCK_SIZE - len(x)
        if pad_len > 0:
            x = torch.cat([x, torch.full((pad_len,), 0, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)])
            
        return x, y

# --- 2. MODEL (Standard) ---
class GPT(nn.Module):
    # ... (Same GPT class as before, just pasting the essentials to save space) ...
    # ... Assume standard nanoGPT implementation here ...
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[
            Block(config) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.token_embedding_table(idx) + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

# Helper classes for GPT (Standard)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T].to(x.device) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        )
    def forward(self, x):
        return x + self.mlp(self.ln2(x + self.attn(self.ln1(x))))

class GPTConfig:
    def __init__(self, vocab_size, block_size, n_layer=6, n_head=6, n_embd=384):
        self.vocab_size = vocab_size; self.block_size = block_size
        self.n_layer = n_layer; self.n_head = n_head; self.n_embd = n_embd

# --- 3. TRAINING LOOP ---
def main():
    dataset = ImaginationDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    config = GPTConfig(vocab_size=dataset.vocab_size, block_size=BLOCK_SIZE)
    model = GPT(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    iter_num = 0
    model.train()
    start = time.time()
    
    print("Starting Training...")
    while iter_num < MAX_ITERS:
        for xb, yb in dataloader:
            if iter_num >= MAX_ITERS: break
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            # CRITICAL CHECK: Max ID
            if iter_num == 0:
                print(f"DEBUG: Max Token ID in batch: {xb.max().item()} (Vocab: {dataset.vocab_size})")
                assert xb.max().item() < dataset.vocab_size, "Token ID overflow!"

            logits, loss = model(xb, yb)
            
            optimizer.zero_grad()
            loss.backward()
            
            # STABILITY FIX: Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # --- PROGRESS SAVING ---
            if iter_num > 0 and iter_num % SAVE_INTERVAL == 0:
                checkpoint_name = f"imaginer_ckpt_{iter_num}.pth"
                torch.save(model.state_dict(), checkpoint_name)
                print(f"--> Saved backup: {checkpoint_name}")
            
            if iter_num % 100 == 0:
                print(f"Iter {iter_num}: Loss {loss.item():.4f}")
            iter_num += 1

    print(f"Done in {time.time()-start:.1f}s")
    torch.save(model.state_dict(), "imaginer_final.pth")

if __name__ == "__main__":
    main()