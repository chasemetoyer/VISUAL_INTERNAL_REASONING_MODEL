import os
import json
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from taming.models.vqgan import VQModel

# --- CONFIG ---
DATA_DIR = "dataset_imagination_balanced"
OUTPUT_FILE = "train_data_2k.pt"
VQGAN_CONFIG = "vqgan_imagenet_f16_16384.yaml"
VQGAN_CKPT = "vqgan_imagenet_f16_16384.ckpt"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_vqgan():
    print("Loading VQGAN...")
    config = yaml.safe_load(open(VQGAN_CONFIG, 'r'))
    model = VQModel(**config['model']['params'])
    state = torch.load(VQGAN_CKPT, map_location=device, weights_only=False)
    # Remove loss/discriminator weights to avoid mismatch (we only need the encoder)
    new_state_dict = {k: v for k, v in state['state_dict'].items() if not k.startswith('loss.')}
    model.load_state_dict(new_state_dict, strict=False)
    model.eval().to(device)
    return model

def preprocess_image(img_path):
    # Load, resize to 256x256, normalize to [-1, 1]
    img = Image.open(img_path).convert("RGB")
    img = img.resize((256, 256))
    img_tensor = torch.tensor(list(img.getdata()), dtype=torch.float32).view(256, 256, 3)
    img_tensor = img_tensor / 127.5 - 1.0  # Normalize to [-1, 1]
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0) # [1, 3, 256, 256]
    return img_tensor.to(device)

def main():
    # 1. Setup Models
    vqgan = load_vqgan()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # 2. Add Special Tokens
    # We add distinct tokens for start/end of the image block
    special_tokens = {'additional_special_tokens': ['<IMG_START>', '<IMG_END>']}
    tokenizer.add_special_tokens(special_tokens)
    
    IMG_START_ID = tokenizer.convert_tokens_to_ids('<IMG_START>')
    IMG_END_ID = tokenizer.convert_tokens_to_ids('<IMG_END>')
    
    # CALCULATE OFFSET
    # Image tokens start AFTER the text vocabulary
    # GPT2 vocab is ~50257. We shift image token 0 to 50257+X
    VOCAB_OFFSET = len(tokenizer) 
    print(f"Text Vocab Size: {VOCAB_OFFSET}")
    print(f"Image Tokens will start at ID: {VOCAB_OFFSET}")
    
    # 3. Process Data
    processed_samples = []
    
    with open(os.path.join(DATA_DIR, "dataset.jsonl"), "r") as f:
        lines = f.readlines()
        
    print(f"Processing {len(lines)} samples...")
    
    for line in tqdm(lines):
        entry = json.loads(line)
        
        # A. Process Text
        prompt_ids = tokenizer.encode(entry['prompt'])
        answer_ids = tokenizer.encode(" " + entry['answer']) # Space is important for GPT tokens
        
        # B. Process Image
        img_tensor = preprocess_image(entry['image_path'])
        with torch.no_grad():
            # Get the codebook indices (0 - 16383)
            # The structure of the return depends on the version
            results = vqgan.encode(img_tensor)
            # Typically returns (z, loss, info)
            # info is a tuple containing indices
            z, loss, info = results
            # info[2] is usually the indices in VectorQuantizer2
            indices = info[2]
            
            # FLATTEN and SHIFT
            # [1, 256] -> [256]
            indices = indices.reshape(-1).cpu().tolist()
            # Shift by the vocabulary offset
            shifted_indices = [idx + VOCAB_OFFSET for idx in indices]
            
        # C. Construct Full Sequence
        # [PROMPT] [IMG_START] [IMAGE] [IMG_END] [ANSWER]
        full_sequence = (
            prompt_ids + 
            [IMG_START_ID] + 
            shifted_indices + 
            [IMG_END_ID] + 
            answer_ids
        )
        
        processed_samples.append(torch.tensor(full_sequence, dtype=torch.long))

    # 4. Save
    print("Saving processed tensors...")
    torch.save({
        "samples": processed_samples,
        "vocab_size_text": VOCAB_OFFSET,
        "vocab_size_image": 16384,
        "vocab_size_total": VOCAB_OFFSET + 16384
    }, OUTPUT_FILE)
    
    print(f"Done! Saved to {OUTPUT_FILE}")
    print(f"Final Vocab Size to use in Model: {VOCAB_OFFSET + 16384}")

if __name__ == "__main__":
    main()