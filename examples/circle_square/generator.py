from PIL import Image, ImageDraw
import random
import os
import csv
import numpy as np

def draw_shapes(top_left_x, top_left_y, bottom_right_x, bottom_right_y,
                    predicted_top_left_x, predicted_top_left_y, predicted_bottom_right_x, predicted_bottom_right_y, center_x, center_y, radius, predicted_center_x, predicted_center_y, predicted_radius, image_size=(128, 128)):
    # Denormalize the coordinates for ground truth
    img = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(img)

    # Draw the ground truth circle
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), outline='red', width=3)

    # Draw the predicted circle
    draw.ellipse((predicted_center_x - predicted_radius, predicted_center_y - predicted_radius, 
                  predicted_center_x + predicted_radius, predicted_center_y + predicted_radius), outline='green', width=3)

    # Draw the ground truth rectangle
    draw.rectangle([(top_left_x, bottom_right_y),(bottom_right_x, top_left_y)], outline='blue', width=3)

    # Draw the predicted rectangle
    draw.rectangle([(predicted_top_left_x, predicted_bottom_right_y), (predicted_bottom_right_x, predicted_top_left_y)], outline='green', width=3)
    return img

def create_image_with_shapes(image_size=(128, 128), circle_radius_range=(10, 19), rect_size_range=(19, 50), line_width=3):
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

    temp = rect_tl[1]

    rect_tl = ( rect_tl[0], rect_br[1])

    rect_br = (rect_br[0], temp)

    return img, circle_center, circle_radius, rect_tl, rect_br

# Function to determine the relationship
def determine_relationship(circle_center, circle_radius, rect_tl, rect_br):
    c1, c2 = circle_center
    r = circle_radius
    t1, t2 = rect_tl
    b1, b2 = rect_br

    # Inside rule
    inside = ((c1 - r) > t1) and ((c1 + r) < b1) and ((c2 - r) > b2) and ((c2 + r) < t2)

    # Outside rule
    outside = ((c1 + r) < t1) or ((c1 - r) > b1) or ((c2 + r) < b2) or ((c2 - r) > t2)

    # Intersect rule
    intersect = not inside and not outside

    if inside:
        return 0
    elif outside:
        return 2
    elif intersect:
        return 1

# Normalize coordinates to be between -1 and 1
def normalize_coordinates(value, max_value):
    return value 

# Generate balanced dataset
def generate_balanced_dataset(num_images_per_class, output_dir='datasets/shape_dataset'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = []
    metadata = []
    counts = {0: 0, 1: 0, 2: 0}
    image_size = (128, 128)

    while min(counts.values()) < num_images_per_class:
        img, circle_center, circle_radius, rect_tl, rect_br = create_image_with_shapes(image_size)
        label = determine_relationship(circle_center, circle_radius, rect_tl, rect_br)
        if counts[label] < num_images_per_class:
            img_path = os.path.join(output_dir, f'image_{len(data)}.png')
            img.save(img_path)
            data.append([img_path, label])

            metadata.append({
                "circle_center_x": normalize_coordinates(circle_center[0], image_size[0]),
                "circle_center_y": normalize_coordinates(circle_center[1], image_size[1]),
                "circle_radius": normalize_coordinates(circle_radius, min(image_size)),
                "rect_tl_x": normalize_coordinates(rect_tl[0], image_size[0]),
                "rect_tl_y": normalize_coordinates(rect_tl[1], image_size[1]),
                "rect_br_x": normalize_coordinates(rect_br[0], image_size[0]),
                "rect_br_y": normalize_coordinates(rect_br[1], image_size[1])
            })
            counts[label] += 1

    # Save metadata to a CSV file
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "circle_center_x", "circle_center_y",
                                               "circle_radius", "rect_tl_x", "rect_tl_y", "rect_br_x", "rect_br_y"])
        writer.writeheader()
        writer.writerows(metadata)

    return data, metadata
