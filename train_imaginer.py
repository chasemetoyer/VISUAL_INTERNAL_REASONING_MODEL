import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import math
import time

# --- CONFIGURATION ---
DATA_PATH = "train_data_2k.pt"
BATCH_SIZE = 8   # If OOM (Out of Memory), lower to 4
BLOCK_SIZE = 512 # Context window (Text + 256 Image + Answer fits easily here)
LEARNING_RATE = 3e-4
MAX_ITERS = 2500  # For 2k samples, 500 iters is enough to overfit check
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# --- 1. THE GPT MODEL (nanoGPT style) ---
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
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
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.act     = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPTConfig:
    def __init__(self, vocab_size, block_size, n_layer=6, n_head=6, n_embd=384):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
        tok_emb = self.token_embedding_table(idx)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten for CrossEntropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

# --- 2. DATA LOADER ---
class ImaginationDataset(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path)
        self.samples = data['samples']
        self.vocab_size = data['vocab_size_total']
        print(f"Loaded dataset with {len(self.samples)} samples. Vocab Size: {self.vocab_size}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # We need to pad or truncate to BLOCK_SIZE
        # In this simplified version, we just assume sequences are < BLOCK_SIZE
        # and pad with 0 if necessary (though 0 is technically a token, for this test it's fine)
        seq = self.samples[idx]
        if len(seq) > BLOCK_SIZE:
            seq = seq[:BLOCK_SIZE]
        
        # Inputs (x) and Targets (y) are shifted by 1
        x = seq[:-1]
        y = seq[1:]
        
        # Pad to fixed length
        pad_len = BLOCK_SIZE - len(x)
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(pad_len, dtype=torch.long)])
            y = torch.cat([y, torch.zeros(pad_len, dtype=torch.long)]) # -100 is standard ignore index
            
        return x, y

# --- 3. TRAINING LOOP ---
def main():
    print(f"Training on {DEVICE}...")
    
    # Load Data
    dataset = ImaginationDataset(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Init Model
    # Small Config for Debugging (6 layers, 384 embedding)
    config = GPTConfig(vocab_size=dataset.vocab_size, block_size=BLOCK_SIZE, 
                       n_layer=6, n_head=6, n_embd=384)
    model = GPT(config)
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    model.train()
    start_time = time.time()
    
    print("\nStarting Training Loop...")
    for iter_num, (xb, yb) in enumerate(dataloader):
        if iter_num >= MAX_ITERS:
            break
            
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        
        # Forward
        logits, loss = model(xb, yb)
        
        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if iter_num % 10 == 0:
            print(f"Iter {iter_num}: Loss {loss.item():.4f}")
            
    print(f"\nTraining Finished in {time.time()-start_time:.2f}s")
    
    # Save Model
    torch.save(model.state_dict(), "imaginer_model.pth")
    print("Model saved to imaginer_model.pth")

if __name__ == "__main__":
    main()