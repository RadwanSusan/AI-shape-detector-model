"""
Generate synthetic shape dataset for training
"""
import os
import random
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm

# Configuration
IMG_SIZE = 128
SHAPES = ['circle', 'square', 'triangle', 'rectangle']
COLORS = [(255, 255, 255), (200, 200, 200), (180, 180, 180)]  # Grayscale variations


def generate_circle(draw, img_size, min_size=30, max_size=55):
    """Generate a circle on the image"""
    size = random.randint(min_size, max_size)
    x = random.randint(size, img_size - size)
    y = random.randint(size, img_size - size)
    color = random.choice(COLORS)

    draw.ellipse([x - size, y - size, x + size, y + size], fill=color, outline=color)


def generate_square(draw, img_size, min_size=30, max_size=55):
    """Generate a square on the image"""
    size = random.randint(min_size, max_size)
    x = random.randint(size, img_size - size)
    y = random.randint(size, img_size - size)
    color = random.choice(COLORS)

    draw.rectangle([x - size, y - size, x + size, y + size], fill=color, outline=color)


def generate_triangle(draw, img_size, min_size=30, max_size=55):
    """Generate a triangle on the image"""
    size = random.randint(min_size, max_size)
    x = random.randint(size, img_size - size)
    y = random.randint(size, img_size - size)
    color = random.choice(COLORS)

    # Equilateral triangle points
    points = [
        (x, y - size),  # Top
        (x - size, y + size),  # Bottom left
        (x + size, y + size)   # Bottom right
    ]
    draw.polygon(points, fill=color, outline=color)


def generate_rectangle(draw, img_size, min_size=20, max_size=35):
    """Generate a rectangle on the image"""
    width = random.randint(min_size, max_size)
    height = random.randint(int(min_size * 1.3), int(max_size * 1.5))
    # Ensure we don't exceed image bounds
    height = min(height, img_size // 2 - 5)
    x = random.randint(width, img_size - width)
    y = random.randint(height, img_size - height)
    color = random.choice(COLORS)

    draw.rectangle([x - width, y - height, x + width, y + height], fill=color, outline=color)


def add_noise(img, noise_level=0.05):
    """Add random noise to image"""
    img_array = np.array(img)
    noise = np.random.randint(-int(255 * noise_level), int(255 * noise_level), img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)


def generate_shape_image(shape_type, img_size=IMG_SIZE):
    """Generate a single shape image"""
    # Create black background
    img = Image.new('RGB', (img_size, img_size), color='black')
    draw = ImageDraw.Draw(img)

    # Draw shape
    if shape_type == 'circle':
        generate_circle(draw, img_size)
    elif shape_type == 'square':
        generate_square(draw, img_size)
    elif shape_type == 'triangle':
        generate_triangle(draw, img_size)
    elif shape_type == 'rectangle':
        generate_rectangle(draw, img_size)

    # Add random noise
    if random.random() > 0.5:
        img = add_noise(img, noise_level=random.uniform(0.02, 0.08))

    # Random rotation
    if random.random() > 0.3:
        angle = random.randint(-30, 30)
        img = img.rotate(angle, fillcolor='black')

    return img


def generate_dataset(num_samples_per_class=1500, output_dir='data'):
    """
    Generate complete dataset

    Args:
        num_samples_per_class: Number of images per shape class
        output_dir: Root directory for dataset
    """
    splits = {
        'train': 0.7,
        'val': 0.15,
        'test': 0.15
    }

    print(f"Generating {num_samples_per_class * len(SHAPES)} total images...")
    print(f"Classes: {SHAPES}")

    for shape in SHAPES:
        print(f"\nGenerating {shape}s...")

        # Calculate split sizes
        train_size = int(num_samples_per_class * splits['train'])
        val_size = int(num_samples_per_class * splits['val'])
        test_size = num_samples_per_class - train_size - val_size

        # Generate for each split
        for split, size in [('train', train_size), ('val', val_size), ('test', test_size)]:
            split_dir = os.path.join(output_dir, split, shape)
            os.makedirs(split_dir, exist_ok=True)

            for i in tqdm(range(size), desc=f"  {split}"):
                img = generate_shape_image(shape)
                img.save(os.path.join(split_dir, f"{shape}_{i:04d}.png"))

    print("\n✓ Dataset generation complete!")
    print(f"  Train: {train_size * len(SHAPES)} images")
    print(f"  Val:   {val_size * len(SHAPES)} images")
    print(f"  Test:  {test_size * len(SHAPES)} images")
    print(f"  Total: {num_samples_per_class * len(SHAPES)} images")


if __name__ == "__main__":
    # Generate dataset with 1500 samples per class (6000 total)
    generate_dataset(num_samples_per_class=1500, output_dir='data')
