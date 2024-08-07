from PIL import Image, ImageDraw
import random
import os
import csv
import numpy as np

def denormalize_coordinates(value, max_value):
    return value * max_value

def draw_shapes(center_x, center_y, radius, top_left_x, top_left_y, bottom_right_x, bottom_right_y,
                predicted_center_x, predicted_center_y, predicted_radius, predicted_top_left_x, predicted_top_left_y, 
                predicted_bottom_right_x, predicted_bottom_right_y, image_size=(128, 128)):
    # Denormalize the coordinates for ground truth
    center_x = denormalize_coordinates(center_x, image_size[0])
    center_y = denormalize_coordinates(center_y, image_size[1])
    radius = denormalize_coordinates(radius, min(image_size))
    top_left_x = denormalize_coordinates(top_left_x, image_size[0])
    top_left_y = denormalize_coordinates(top_left_y, image_size[1])
    bottom_right_x = denormalize_coordinates(bottom_right_x, image_size[0])
    bottom_right_y = denormalize_coordinates(bottom_right_y, image_size[1])

    # Denormalize the coordinates for prediction
    predicted_center_x = denormalize_coordinates(predicted_center_x, image_size[0])
    predicted_center_y = denormalize_coordinates(predicted_center_y, image_size[1])
    predicted_radius = denormalize_coordinates(predicted_radius, min(image_size))
    predicted_top_left_x = denormalize_coordinates(predicted_top_left_x, image_size[0])
    predicted_top_left_y = denormalize_coordinates(predicted_top_left_y, image_size[1])
    predicted_bottom_right_x = denormalize_coordinates(predicted_bottom_right_x, image_size[0])
    predicted_bottom_right_y = denormalize_coordinates(predicted_bottom_right_y, image_size[1])

    img = Image.new('RGB', image_size, color='white')
    draw = ImageDraw.Draw(img)

    # Draw the ground truth shapes
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), outline='red', width=3)
    draw.rectangle([(top_left_x, top_left_y), (bottom_right_x, bottom_right_y)], outline='blue', width=3)

    # Draw the predicted shapes
    draw.ellipse((predicted_center_x - predicted_radius, predicted_center_y - predicted_radius, 
                  predicted_center_x + predicted_radius, predicted_center_y + predicted_radius), outline='green', width=3)
    draw.rectangle([(predicted_top_left_x, predicted_top_left_y), 
                    (predicted_bottom_right_x, predicted_bottom_right_y)], outline='purple', width=3)

    return img

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

# Normalize coordinates to be between -1 and 1
def normalize_coordinates(value, max_value):
    return (value / max_value) 

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
