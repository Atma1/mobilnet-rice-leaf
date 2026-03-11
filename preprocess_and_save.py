import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
from PIL import Image

# Path to the input and output images
input_path = '/workspaces/mobilnet-rice-leaf/work/dataset/bacterial_leaf_blight/IMG_20231018_144114.jpg'
output_path = 'pre-processed.jpeg'

# MobileNetV3 expects 224x224 images by default
IMG_SIZE = 224

# Load image
img = Image.open(input_path).convert('RGB')
img = img.resize((IMG_SIZE, IMG_SIZE))

# Convert to numpy array and add batch dimension
img_array = np.array(img, dtype=np.float32)
img_array = np.expand_dims(img_array, axis=0)

# Preprocess using MobileNetV3's preprocess_input
preprocessed = preprocess_input(img_array)

# Remove batch dimension and convert back to uint8 for saving
preprocessed_img = np.squeeze(preprocessed, axis=0)
preprocessed_img = np.clip(preprocessed_img, 0, 255).astype(np.uint8)

# Save the preprocessed image
Image.fromarray(preprocessed_img).save(output_path, format='JPEG')

print(f'Preprocessed image saved to {output_path}')

model = tf.keras.models.load_model("/workspaces/mobilnet-rice-leaf/work/models/model_scenario_02_90-10_SGD_lr0.0005.keras")
preds = model(preprocessed)
print(preds)
