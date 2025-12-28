import torch
import yaml
import numpy as np
from PIL import Image
from taming.models.vqgan import VQModel
from transformers import GPT2TokenizerFast
from train_imaginer import GPT, GPTConfig # Import your model definition

# --- CONFIG ---
MODEL_PATH = "imaginer_model.pth"
VQGAN_CONFIG = "vqgan_imagenet_f16_16384.yaml"
VQGAN_CKPT = "vqgan_imagenet_f16_16384.ckpt"
DEVICE = 'cpu' # Keep on CPU for inference to match your setup

def load_vqgan():
    config = yaml.safe_load(open(VQGAN_CONFIG, 'r'))
    model = VQModel(**config['model']['params'])
    state = torch.load(VQGAN_CKPT, map_location=DEVICE)
    model.load_state_dict(state['state_dict'], strict=False)
    model.eval()
    return model

def generate_imagination(prompt_text):
    # 1. Load Everything
    print("Loading models...")
    
    # Load Tokenizer & VQGAN
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'additional_special_tokens': ['<IMG_START>', '<IMG_END>']})
    vqgan = load_vqgan()
    
    # Load Your Trained Model
    # Note: We must match the vocab size from training (Saved in the print logs)
    # If you forgot it, it's 50257 (GPT2) + 2 (Special) + 16384 (Image) = 66643
    VOCAB_SIZE = 66643 
    config = GPTConfig(vocab_size=VOCAB_SIZE, block_size=512, n_layer=6, n_head=6, n_embd=384)
    model = GPT(config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Prepare Input
    print(f"Dreaming about: '{prompt_text}'")
    prompt_ids = tokenizer.encode(prompt_text)
    IMG_START_ID = tokenizer.convert_tokens_to_ids('<IMG_START>')
    IMG_END_ID = tokenizer.convert_tokens_to_ids('<IMG_END>')
    
    # Start sequence: [Prompt] [IMG_START]
    input_ids = torch.tensor(prompt_ids + [IMG_START_ID], dtype=torch.long).unsqueeze(0).to(DEVICE)

    # 3. Generate Visual Tokens
    # We want exactly 256 image tokens (16x16 grid)
    print("Generating visual tokens...")
    generated_img_tokens = []
    
    with torch.no_grad():
        for _ in range(256):
            logits, _ = model(input_ids)
            # Take logits from the last step only
            logits = logits[:, -1, :] 
            # Greedy decode (pick the most likely pixel)
            next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
            
            input_ids = torch.cat((input_ids, next_token), dim=1)
            generated_img_tokens.append(next_token.item())

    # 4. Decode Image
    # Shift tokens back to VQGAN range (0-16383)
    # Offset = Text Vocab (50257) + Special (2) = 50259
    OFFSET = 50259 
    
    # Filter out any tokens that might be outside valid range (hallucinations)
    valid_tokens = []
    for t in generated_img_tokens:
        if t >= OFFSET and t < OFFSET + 16384:
            valid_tokens.append(t - OFFSET)
        else:
            # If model glitched and predicted a text word, replace with 0 (black)
            valid_tokens.append(0) 

    z_indices = torch.tensor(valid_tokens, dtype=torch.long).to(DEVICE)
    
    # VQGAN needs shape [1, 256]
    z_indices = z_indices.unsqueeze(0) 
    
    # Decode to pixels
    x_recon = vqgan.decode(vqgan.quantize.get_codebook_entry(z_indices, shape=(1, 16, 16, 256)))
    
    # Save Image
    # Convert [-1, 1] -> [0, 255]
    x_recon = torch.clamp(x_recon, -1., 1.)
    x_recon = (x_recon + 1.0) / 2.0 * 255.0
    x_recon = x_recon.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    
    img = Image.fromarray(x_recon[0])
    img.save("imagination_result.png")
    print("Saved 'imagination_result.png'")

if __name__ == "__main__":
    # Test with a prompt from your dataset
    TEST_PROMPT = "Imagine a red square at grid (2, 2) and a blue circle at (5, 5). Move red square 2 right, 0 down. Overlap?"
    generate_imagination(TEST_PROMPT)