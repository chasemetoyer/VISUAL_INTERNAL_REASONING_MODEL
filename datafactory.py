import os
import random
import json
import numpy as np
from PIL import Image, ImageDraw

# --- CONFIGURATION ---
DATASET_SIZE = 1000  # Start small for testing, scale to 100k later
OUTPUT_DIR = "dataset_imagination_v1"
IMG_SIZE = 256
GRID_SCALE = 10  # 10x10 logical grid mapped to 256x256 pixels
OBJ_SIZE = 20    # Radius or half-width of objects (in pixels)

# Colors and Shapes
COLORS = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 128, 0),
    "yellow": (255, 255, 0)
}
SHAPES = ["circle", "square"]

class Shape:
    def __init__(self, color_name, shape_type, x, y):
        self.color_name = color_name
        self.color_rgb = COLORS[color_name]
        self.type = shape_type
        self.x = x  # Center X (pixels)
        self.y = y  # Center Y (pixels)

    def get_bbox(self):
        return [self.x - OBJ_SIZE, self.y - OBJ_SIZE, 
                self.x + OBJ_SIZE, self.y + OBJ_SIZE]

def add_visual_noise(draw):
    """Adds random lines/dots so the model can't memorize pixel indices."""
    for _ in range(random.randint(5, 15)):
        x1, y1 = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
        x2, y2 = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
        fill = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        draw.line([(x1, y1), (x2, y2)], fill=fill, width=1)

def render_scene(shapes, filename):
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
    draw = ImageDraw.Draw(img)
    
    # 1. Add Visual Noise (Anti-Cheat)
    add_visual_noise(draw)
    
    # 2. Draw Shapes
    for s in shapes:
        bbox = s.get_bbox()
        if s.type == "circle":
            draw.ellipse(bbox, fill=s.color_rgb, outline="black")
        elif s.type == "square":
            draw.rectangle(bbox, fill=s.color_rgb, outline="black")
            
    img.save(filename)
    return img

def check_overlap(s1, s2):
    """Simple bounding box overlap logic."""
    # Euclidian distance check is cleaner for "touching" logic
    dist = np.sqrt((s1.x - s2.x)**2 + (s1.y - s2.y)**2)
    # If distance < sum of radii (approx), they overlap
    return dist < (OBJ_SIZE * 1.8) # 1.8 gives a little buffer

def generate_sample(sample_id):
    # 1. Setup Logic
    # We work in "Grid Units" (0-10) to make text descriptions clean, 
    # but render in pixels with jitter.
    
    # Object A (The Mover)
    grid_x_start = random.randint(1, 8)
    grid_y_start = random.randint(1, 8)
    
    # Object B (The Anchor - Stationary)
    grid_x_anchor = random.randint(1, 8)
    grid_y_anchor = random.randint(1, 8)
    
    # Define Move (Delta)
    dx = random.choice([-2, -1, 0, 1, 2])
    dy = random.choice([-2, -1, 0, 1, 2])
    
    # Ensure we actually move somewhere
    if dx == 0 and dy == 0: dx = 1
        
    # Create Objects
    # Add random pixel jitter (+/- 5px) so strict grid memorization fails
    jitter = lambda: random.randint(-5, 5)
    
    unit = IMG_SIZE // GRID_SCALE # 25 pixels per grid unit
    
    pixel_x_start = (grid_x_start * unit) + jitter()
    pixel_y_start = (grid_y_start * unit) + jitter()
    
    pixel_x_anchor = (grid_x_anchor * unit) + jitter()
    pixel_y_anchor = (grid_y_anchor * unit) + jitter()
    
    obj_mover = Shape("red", "square", pixel_x_start, pixel_y_start)
    obj_anchor = Shape("blue", "circle", pixel_x_anchor, pixel_y_anchor)
    
    # 2. Text Prompt Generation
    prompt = (f"Imagine a red square at grid position ({grid_x_start}, {grid_y_start}) "
              f"and a blue circle at ({grid_x_anchor}, {grid_y_anchor}). "
              f"Move the red square {dx} units right and {dy} units down. "
              f"Does it overlap with the blue circle?")
    
    # 3. Simulate the Move (The "Ground Truth" Logic)
    final_x = pixel_x_start + (dx * unit)
    final_y = pixel_y_start + (dy * unit)
    
    # Update Mover Position for Rendering
    obj_mover.x = final_x
    obj_mover.y = final_y
    
    # 4. Render the "Imagined" State
    # Note: We verify if the move put it off-screen, but we render it anyway
    img_filename = os.path.join(OUTPUT_DIR, "images", f"img_{sample_id:06d}.png")
    render_scene([obj_anchor, obj_mover], img_filename)
    
    # 5. Compute Answer
    is_overlapping = check_overlap(obj_mover, obj_anchor)
    answer = "Yes" if is_overlapping else "No"
    
    return {
        "id": sample_id,
        "prompt": prompt,
        "image_path": img_filename,
        "answer": answer,
        "metadata": {
            "dx": dx, "dy": dy, "start": (grid_x_start, grid_y_start)
        }
    }

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Create directories
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
    
    metadata_file = os.path.join(OUTPUT_DIR, "dataset.jsonl")
    
    print(f"Generating {DATASET_SIZE} samples...")
    
    with open(metadata_file, "w") as f:
        for i in range(DATASET_SIZE):
            sample = generate_sample(i)
            f.write(json.dumps(sample) + "\n")
            
            if i % 100 == 0:
                print(f"Generated {i} samples...", end="\r")
                
    print(f"\nDone! Data saved to {OUTPUT_DIR}")