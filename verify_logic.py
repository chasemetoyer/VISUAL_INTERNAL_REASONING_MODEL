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

def check_logic():
    # 1. Load Data & Model
    data = torch.load(DATA_PATH)
    meta = data['meta']
    vocab_size = meta['vocab_size_total']
    img_start_id = meta['img_start_id']
    img_end_id = meta['img_end_id']
    
    config = GPTConfig(vocab_size=vocab_size, block_size=512)
    model = GPT(config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval().to(DEVICE)
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # 2. The Test Case
    # Correct Answer: NO (Red ends at 4,2; Blue is at 6,6)
    prompt = "Imagine a red square at grid (2, 2) and a blue circle at (6, 6). Move red square 2 right, 0 down. Overlap?"
    print(f"QUESTION: {prompt}")
    
    input_ids = tokenizer.encode(prompt) + [img_start_id]
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    # 3. Generate (Fast Loop)
    print("Thinking...", end="")
    with torch.no_grad():
        # Stop when we hit a newline or answer
        for i in range(300): # Allow 256 image + 10 text tokens
            logits, _ = model(input_ids)
            logits = logits[:, -1, :]
            
            # Greedy decode for the answer (we want the most likely logic)
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            
            # Print progress dots
            if i % 50 == 0: print(".", end="", flush=True)
            
            # If we passed the image block, start looking for text
            if next_token.item() == img_end_id:
                print(" [Image Complete] ", end="")
            
            # If we see the tokenizer's "End of Text" or similar, break
            # For now, just print the decoded text at the end
    
    # 4. Decode the text AFTER the image
    full_sequence = input_ids[0].cpu().tolist()
    
    try:
        # Find where image ends
        end_pos = full_sequence.index(img_end_id)
        answer_tokens = full_sequence[end_pos+1:]
        answer_text = tokenizer.decode(answer_tokens)
        print(f"\n\nMODEL ANSWER: {answer_text}")
    except ValueError:
        print("\n\nError: Model never finished the image block.")

if __name__ == "__main__":
    check_logic()