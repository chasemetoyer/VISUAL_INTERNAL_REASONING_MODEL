import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from train_imaginer import GPT, GPTConfig 

# --- CONFIG ---
MODEL_PATH = "imaginer_final.pth"
DATA_PATH = "train_data_final.pt"

if torch.cuda.is_available(): DEVICE = 'cuda'
elif torch.backends.mps.is_available(): DEVICE = 'mps'
else: DEVICE = 'cpu'

def blindfold_test():
    # 1. Load Data & Model
    data = torch.load(DATA_PATH)
    meta = data['meta']
    vocab_size = meta['vocab_size_total']
    img_start_id = meta['img_start_id']
    img_end_id = meta['img_end_id']
    vocab_offset = meta['vocab_offset']
    
    config = GPTConfig(vocab_size=vocab_size, block_size=512)
    model = GPT(config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    prompt = "Imagine a red square at grid (2, 2) and a blue circle at (6, 6). Move red square 2 right, 0 down. Overlap?"
    print(f"PROMPT: {prompt}")
    
    input_ids = tokenizer.encode(prompt) + [img_start_id]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # 2. Generate Image (Normal)
    print("Generating Image...", end="")
    image_tokens = []
    with torch.no_grad():
        for i in range(256):
            logits, _ = model(input_ids)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            image_tokens.append(next_token)
    print(" Done.")

    # 3. THE BLINDFOLD INTERVENTION
    # We take the sequence, but we CORRUPT the image part
    # Sequence is: [Prompt] [START] [256 Image Tokens]
    # We replace [256 Image Tokens] with Random Integers
    
    print("\n--- APPLYING BLINDFOLD (Replacing dream with noise) ---")
    
    # Create random noise in the valid visual range
    noise = torch.randint(vocab_offset, vocab_offset + 16384, (1, 256)).to(DEVICE)
    
    # Reconstruct the sequence: Prompt + START + NOISE
    prompt_tensor = input_ids[:, : -256] # Keep prompt and start
    blindfolded_input = torch.cat((prompt_tensor, noise), dim=1)
    
    # Add <IMG_END> manually so the model knows to start answering
    end_token = torch.tensor([[img_end_id]]).to(DEVICE)
    blindfolded_input = torch.cat((blindfolded_input, end_token), dim=1)

    # 4. Ask Model to Answer based on Noise
    print("Asking model to answer based on the noise...")
    output = blindfolded_input
    
    with torch.no_grad():
        for _ in range(10): # Generate 10 text tokens
            logits, _ = model(output)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            output = torch.cat((output, next_token), dim=1)
            
            # Stop if we see end of text (optional, simplifed here)

    # Decode response
    # We only care about what came AFTER the noise and END token
    answer_tokens = output[0, -10:].tolist() # Just grab the last few
    text = tokenizer.decode(answer_tokens)
    print(f"\nBLINDFOLDED ANSWER: {text}")

if __name__ == "__main__":
    blindfold_test()