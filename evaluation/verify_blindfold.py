import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train_imaginer import GPT, GPTConfig 

# --- CONFIG ---
MODEL_PATH = "imaginer_ckpt_15000.pth"
DATA_PATH = "train_data_final.pt"
IMG_TOKENS = 256
ANSWER_TOKENS = 10
IGNORE_VAL = -1e10

if torch.cuda.is_available(): DEVICE = 'cuda'
elif torch.backends.mps.is_available(): DEVICE = 'mps'
else: DEVICE = 'cpu'

def blindfold_test_strict():
    # 1. Load Data & Model
    data = torch.load(DATA_PATH, weights_only=False)
    meta = data['meta']
    vocab_size = meta['vocab_size_total']
    img_start_id = meta['img_start_id']
    img_end_id = meta['img_end_id']
    vocab_offset = meta['vocab_offset']
    
    config = GPTConfig(vocab_size=vocab_size, block_size=512)
    model = GPT(config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval().to(DEVICE)
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    prompt = "Imagine a red square at grid (2, 2) and a blue circle at (6, 6). Move red square 2 right, 0 down. Overlap?"
    print(f"PROMPT: {prompt}")
    
    input_ids = tokenizer.encode(prompt) + [img_start_id]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # 2. Generate Image (Normal)
    print("Generating Image (Internal)...", end="")
    with torch.no_grad():
        for _ in range(IMG_TOKENS):
            logits, _ = model(input_ids)
            logits = logits[:, -1, :].clone()
            logits[:, :vocab_offset] = IGNORE_VAL # Force visual tokens
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat((input_ids, next_token), dim=1)
    print(" Done.")

    # 3. APPLY BLINDFOLD
    print("\n--- APPLYING BLINDFOLD ---")
    
    # Generate Valid VQGAN Noise
    # Must be between vocab_offset and vocab_size
    noise = torch.randint(vocab_offset, vocab_size, (1, IMG_TOKENS)).to(DEVICE)
    
    # Reconstruct: [Prompt] [START] [NOISE]
    prompt_tensor = input_ids[:, : -IMG_TOKENS] # Strip the real dream
    blindfolded_input = torch.cat((prompt_tensor, noise), dim=1)
    
    # Add <IMG_END> manually
    end_token = torch.tensor([[img_end_id]]).to(DEVICE)
    blindfolded_input = torch.cat((blindfolded_input, end_token), dim=1)

    # 4. Strict Answer Generation
    print("Asking model to answer based on noise...")
    output = blindfolded_input
    with torch.no_grad():
        for _ in range(ANSWER_TOKENS):
            logits, _ = model(output)
            logits = logits[:, -1, :].clone()
            
            # BAN IMAGE TOKENS
            logits[:, vocab_offset:] = IGNORE_VAL
            logits[:, img_start_id] = IGNORE_VAL
            
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            output = torch.cat((output, next_token), dim=1)

    # 5. Decode
    seq = output[0].tolist()
    # Find the manual IMG_END we added
    # It is at index: len(prompt_tensor) + IMG_TOKENS
    end_pos = blindfolded_input.shape[1] - 1
    
    answer_tokens = seq[end_pos+1:]
    answer_text = tokenizer.decode(answer_tokens)
    print(f"\nBLINDFOLDED ANSWER: {answer_text}")

if __name__ == "__main__":
    blindfold_test_strict()