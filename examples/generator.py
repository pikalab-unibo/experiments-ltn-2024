from PIL import Image, ImageDraw
import random
import os

def create_image_with_shapes(image_size=(128, 128), circle_radius_range=(10, 30), rect_size_range=(20, 50), line_width=3):
    img = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Generate random circle
    circle_radius = random.randint(*circle_radius_range)
    circle_center = (random.randint(circle_radius, image_size[0] - circle_radius),
                     random.randint(circle_radius, image_size[1] - circle_radius))
    
    # Draw circle in red with thick outline
    draw.ellipse((circle_center[0] - circle_radius, circle_center[1] - circle_radius,
                  circle_center[0] + circle_radius, circle_center[1] + circle_radius), outline='red', width=line_width)
    
    # Generate random rectangle
    rect_width = random.randint(*rect_size_range)
    rect_height = random.randint(*rect_size_range)
    
    # Ensure rectangle is within image bounds and centered as much as possible
    rect_tl = (random.randint(0, image_size[0] - rect_width),
               random.randint(0, image_size[1] - rect_height))
    rect_br = (rect_tl[0] + rect_width, rect_tl[1] + rect_height)
    
    # Draw rectangle in blue with thick outline
    draw.rectangle([rect_tl, rect_br], outline='blue', width=line_width)
    
    return img, circle_center, circle_radius, rect_tl, rect_br

# Function to determine the relationship
def determine_relationship(circle_center, circle_radius, rect_tl, rect_br):
    # Bounding box of the circle
    circle_tl = (circle_center[0] - circle_radius, circle_center[1] - circle_radius)
    circle_br = (circle_center[0] + circle_radius, circle_center[1] + circle_radius)
    
    # Check inside
    if (circle_tl[0] >= rect_tl[0] and circle_tl[1] >= rect_tl[1] and
        circle_br[0] <= rect_br[0] and circle_br[1] <= rect_br[1]):
        return 0
    
    # Check outside
    if (circle_br[0] < rect_tl[0] or circle_br[1] < rect_tl[1] or
        circle_tl[0] > rect_br[0] or circle_tl[1] > rect_br[1]):
        return 2
    
    # Otherwise, it's overlapping
    return 1

# Generate balanced dataset
def generate_balanced_dataset(num_images_per_class, output_dir='datasets/shape_dataset'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    data = []
    counts = {0: 0, 1: 0, 2: 0}
    
    while min(counts.values()) < num_images_per_class:
        img, circle_center, circle_radius, rect_tl, rect_br = create_image_with_shapes()
        label = determine_relationship(circle_center, circle_radius, rect_tl, rect_br)
        
        if counts[label] < num_images_per_class:
            img_path = os.path.join(output_dir, f'image_{len(data)}.png')
            img.save(img_path)
            data.append((img_path, label))
            counts[label] += 1
    
    return data
