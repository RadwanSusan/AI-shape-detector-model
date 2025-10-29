"""
Advanced synthetic shape dataset generator with robust variations
Generates diverse, challenging images to train a production-ready model
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from tqdm import tqdm

# Configuration
IMG_SIZE = 128
SHAPES = ['circle', 'square', 'triangle', 'rectangle']


def random_color():
    """Generate random RGB color"""
    return tuple(random.randint(50, 255) for _ in range(3))


def random_background(img_size, complexity='medium'):
    """
    Generate various background types

    Args:
        img_size: Image dimension
        complexity: 'simple', 'medium', 'complex'
    """
    bg_type = random.choice(['solid', 'gradient', 'noisy', 'textured'])

    if complexity == 'simple':
        bg_type = 'solid'
    elif complexity == 'complex':
        bg_type = random.choice(['gradient', 'noisy', 'textured'])

    if bg_type == 'solid':
        # Solid color background
        color = tuple(random.randint(0, 100) for _ in range(3))  # Darker backgrounds
        return Image.new('RGB', (img_size, img_size), color)

    elif bg_type == 'gradient':
        # Gradient background
        img = Image.new('RGB', (img_size, img_size))
        draw = ImageDraw.Draw(img)

        # Random gradient direction
        if random.random() > 0.5:
            # Horizontal gradient
            for x in range(img_size):
                ratio = x / img_size
                r = int(random.randint(0, 80) * (1 - ratio) + random.randint(0, 80) * ratio)
                g = int(random.randint(0, 80) * (1 - ratio) + random.randint(0, 80) * ratio)
                b = int(random.randint(0, 80) * (1 - ratio) + random.randint(0, 80) * ratio)
                draw.line([(x, 0), (x, img_size)], fill=(r, g, b))
        else:
            # Vertical gradient
            for y in range(img_size):
                ratio = y / img_size
                r = int(random.randint(0, 80) * (1 - ratio) + random.randint(0, 80) * ratio)
                g = int(random.randint(0, 80) * (1 - ratio) + random.randint(0, 80) * ratio)
                b = int(random.randint(0, 80) * (1 - ratio) + random.randint(0, 80) * ratio)
                draw.line([(0, y), (img_size, y)], fill=(r, g, b))

        return img

    elif bg_type == 'noisy':
        # Noisy background
        base_color = random.randint(0, 60)
        noise = np.random.randint(-30, 30, (img_size, img_size, 3))
        bg_array = np.full((img_size, img_size, 3), base_color) + noise
        bg_array = np.clip(bg_array, 0, 255).astype(np.uint8)
        return Image.fromarray(bg_array)

    else:  # textured
        # Textured background with patterns
        img = Image.new('RGB', (img_size, img_size))
        draw = ImageDraw.Draw(img)

        # Random pattern
        base_color = tuple(random.randint(0, 60) for _ in range(3))
        img.paste(base_color, (0, 0, img_size, img_size))

        # Add random lines or dots
        for _ in range(random.randint(5, 15)):
            if random.random() > 0.5:
                x1, y1 = random.randint(0, img_size), random.randint(0, img_size)
                x2, y2 = random.randint(0, img_size), random.randint(0, img_size)
                color = tuple(random.randint(0, 80) for _ in range(3))
                draw.line([(x1, y1), (x2, y2)], fill=color, width=1)
            else:
                x, y = random.randint(0, img_size), random.randint(0, img_size)
                r = random.randint(2, 8)
                color = tuple(random.randint(0, 80) for _ in range(3))
                draw.ellipse([x-r, y-r, x+r, y+r], fill=color)

        return img


def generate_circle(draw, img_size, min_size=30, max_size=55):
    """Generate a circle with random color"""
    size = random.randint(min_size, max_size)
    x = random.randint(size, img_size - size)
    y = random.randint(size, img_size - size)
    color = random_color()

    # Optional: fill and outline with different colors
    if random.random() > 0.7:
        outline = random_color()
        width = random.randint(2, 4)
        draw.ellipse([x - size, y - size, x + size, y + size], fill=color, outline=outline, width=width)
    else:
        draw.ellipse([x - size, y - size, x + size, y + size], fill=color)


def generate_square(draw, img_size, min_size=30, max_size=55):
    """Generate a square with random color"""
    size = random.randint(min_size, max_size)
    x = random.randint(size, img_size - size)
    y = random.randint(size, img_size - size)
    color = random_color()

    if random.random() > 0.7:
        outline = random_color()
        width = random.randint(2, 4)
        draw.rectangle([x - size, y - size, x + size, y + size], fill=color, outline=outline, width=width)
    else:
        draw.rectangle([x - size, y - size, x + size, y + size], fill=color)


def generate_triangle(draw, img_size, min_size=30, max_size=55):
    """Generate a triangle with random color"""
    size = random.randint(min_size, max_size)
    x = random.randint(size, img_size - size)
    y = random.randint(size, img_size - size)
    color = random_color()

    # Equilateral triangle points
    points = [
        (x, y - size),  # Top
        (x - size, y + size),  # Bottom left
        (x + size, y + size)   # Bottom right
    ]

    if random.random() > 0.7:
        outline = random_color()
        width = random.randint(2, 4)
        draw.polygon(points, fill=color, outline=outline, width=width)
    else:
        draw.polygon(points, fill=color)


def generate_rectangle(draw, img_size, min_size=20, max_size=35):
    """Generate a rectangle with random color"""
    width = random.randint(min_size, max_size)
    height = random.randint(int(min_size * 1.3), int(max_size * 1.5))
    height = min(height, img_size // 2 - 5)
    x = random.randint(width, img_size - width)
    y = random.randint(height, img_size - height)
    color = random_color()

    if random.random() > 0.7:
        outline = random_color()
        width_outline = random.randint(2, 4)
        draw.rectangle([x - width, y - height, x + width, y + height],
                      fill=color, outline=outline, width=width_outline)
    else:
        draw.rectangle([x - width, y - height, x + width, y + height], fill=color)


def add_noise(img, noise_type='gaussian', intensity='medium'):
    """
    Add various types of noise to image

    Args:
        img: PIL Image
        noise_type: 'gaussian', 'salt_pepper', 'both'
        intensity: 'light', 'medium', 'heavy'
    """
    img_array = np.array(img)

    # Set intensity levels
    if intensity == 'light':
        gaussian_sigma = 10
        sp_amount = 0.01
    elif intensity == 'heavy':
        gaussian_sigma = 30
        sp_amount = 0.05
    else:  # medium
        gaussian_sigma = 20
        sp_amount = 0.02

    # Gaussian noise
    if noise_type in ['gaussian', 'both']:
        noise = np.random.normal(0, gaussian_sigma, img_array.shape)
        img_array = img_array + noise

    # Salt and pepper noise
    if noise_type in ['salt_pepper', 'both']:
        mask = np.random.rand(*img_array.shape[:2])
        img_array[mask < sp_amount/2] = 0  # Salt (black)
        img_array[mask > 1 - sp_amount/2] = 255  # Pepper (white)

    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)


def add_blur(img, blur_type='gaussian', intensity='medium'):
    """Add blur effects"""
    if intensity == 'light':
        radius = 1
    elif intensity == 'heavy':
        radius = 4
    else:  # medium
        radius = 2

    if blur_type == 'gaussian':
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    elif blur_type == 'box':
        return img.filter(ImageFilter.BoxBlur(radius=radius))
    else:  # motion - simulate with directional blur
        return img.filter(ImageFilter.BLUR)


def adjust_contrast_brightness(img, contrast_factor=None, brightness_factor=None):
    """Adjust contrast and brightness"""
    if contrast_factor is None:
        contrast_factor = random.uniform(0.5, 1.5)
    if brightness_factor is None:
        brightness_factor = random.uniform(0.6, 1.4)

    # Adjust contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)

    # Adjust brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)

    return img


def generate_shape_image(shape_type, img_size=IMG_SIZE, difficulty='medium'):
    """
    Generate a single shape image with various augmentations

    Args:
        shape_type: Type of shape to generate
        img_size: Image size
        difficulty: 'easy', 'medium', 'hard' - controls augmentation intensity
    """
    # Create background
    if difficulty == 'easy':
        img = Image.new('RGB', (img_size, img_size), color='black')
    else:
        complexity = 'simple' if difficulty == 'medium' else 'complex'
        img = random_background(img_size, complexity=complexity)

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

    # Apply augmentations based on difficulty
    if difficulty in ['medium', 'hard']:
        # Random rotation (full 360 degrees)
        if random.random() > 0.2:
            angle = random.randint(0, 360)
            # Get a fill color from the background
            bg_color = img.getpixel((0, 0))
            img = img.rotate(angle, fillcolor=bg_color, expand=False)

        # Add noise
        if random.random() > 0.4:
            noise_intensity = 'light' if difficulty == 'medium' else random.choice(['medium', 'heavy'])
            noise_type = random.choice(['gaussian', 'salt_pepper', 'both'])
            img = add_noise(img, noise_type=noise_type, intensity=noise_intensity)

        # Add blur
        if random.random() > 0.5:
            blur_intensity = 'light' if difficulty == 'medium' else random.choice(['medium', 'heavy'])
            blur_type = random.choice(['gaussian', 'box', 'motion'])
            img = add_blur(img, blur_type=blur_type, intensity=blur_intensity)

        # Adjust contrast/brightness
        if difficulty == 'hard' and random.random() > 0.3:
            img = adjust_contrast_brightness(img)

    return img


def generate_dataset(num_samples_per_class=10000, output_dir='data', difficulty_mix=True):
    """
    Generate complete dataset with various difficulty levels

    Args:
        num_samples_per_class: Number of images per shape class
        output_dir: Root directory for dataset
        difficulty_mix: If True, mix easy/medium/hard samples
    """
    splits = {
        'train': 0.7,
        'val': 0.15,
        'test': 0.15
    }

    print(f"Generating {num_samples_per_class * len(SHAPES)} total images...")
    print(f"Classes: {SHAPES}")
    print(f"Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Difficulty mix: {difficulty_mix}")

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
                # Determine difficulty
                if difficulty_mix:
                    if split == 'train':
                        # Training: Mix of all difficulties (more medium/hard)
                        difficulty = random.choices(
                            ['easy', 'medium', 'hard'],
                            weights=[0.2, 0.5, 0.3]
                        )[0]
                    else:
                        # Val/Test: Balanced mix
                        difficulty = random.choice(['easy', 'medium', 'hard'])
                else:
                    difficulty = 'medium'

                img = generate_shape_image(shape, difficulty=difficulty)
                img.save(os.path.join(split_dir, f"{shape}_{i:05d}.png"))

    print("\n✓ Dataset generation complete!")
    print(f"  Train: {train_size * len(SHAPES)} images")
    print(f"  Val:   {val_size * len(SHAPES)} images")
    print(f"  Test:  {test_size * len(SHAPES)} images")
    print(f"  Total: {num_samples_per_class * len(SHAPES)} images")


if __name__ == "__main__":
    # Generate advanced dataset with 10,000 samples per class (40K total)
    # Mix of easy, medium, and hard samples for robustness
    generate_dataset(
        num_samples_per_class=10000,
        output_dir='data',
        difficulty_mix=True
    )
