# coding: utf-8

import os
import numpy as np
import rasterio
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from skimage.morphology import skeletonize, dilation, square
from skimage.measure import label
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def skeletonize_and_buffer(mask, buffer_size=3):
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)  # Remove channel dimension for processing
    skeleton = skeletonize(mask.astype(np.uint8))  # Ensure input is NumPy and uint8
    buffered_mask = dilation(skeleton, square(buffer_size))
    
    if mask.ndim == 3:  # If the input had a channel, add it back
        buffered_mask = np.expand_dims(buffered_mask, axis=-1)
    
    return buffered_mask

# def normalize_image(image):
#     min_val = np.min(image)
#     max_val = np.max(image)
#     normalized_image = (image - min_val) / (max_val - min_val)
#     return normalized_image

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val == min_val:
        # Avoid division by zero by setting constant or zeros
        print("Warning: Image has constant pixel values")
        normalized_image = np.zeros_like(image)  # Or set to another constant value
    else:
        normalized_image = (image - min_val) / (max_val - min_val)

    # Replace any NaN values with 0
    normalized_image = np.nan_to_num(normalized_image, nan=0.0)

    return normalized_image


def remove_small_objects(mask, min_size=200):
    labeled_mask, num_features = label(mask, return_num=True)
    small_objects_mask = np.zeros_like(mask)
    
    for i in range(1, num_features + 1):
        component = np.where(labeled_mask == i, 1, 0)
        if np.sum(component) < min_size:
            small_objects_mask += component
    
    return mask - small_objects_mask

class DataGenerator(Sequence):
    def __init__(self, image_list, mask_list, batch_size=32, image_size=(256, 256), shuffle=True, augment=False, min_area=500, buffer_size=3, threshold=75):
        self.image_list = image_list
        self.mask_list = mask_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.min_area = min_area
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.indices = np.arange(len(self.image_list))
        self.image_list, self.mask_list = self.validate_image_dimensions()
        self.indices = np.arange(len(self.image_list))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.image_list) // self.batch_size

    def validate_image_dimensions(self):
        valid_image_list = []
        valid_mask_list = []
        required_size = self.image_size[0]  # Assuming image_size is a square (width, height)
        for img_path, mask_path in zip(self.image_list, self.mask_list):
            with rasterio.open(img_path) as src:
                if src.shape[0] < required_size or src.shape[1] < required_size:
                    continue
            valid_image_list.append(img_path)
            valid_mask_list.append(mask_path)
        print(f"Validated {len(valid_image_list)} images and {len(valid_mask_list)} masks.")
        return valid_image_list, valid_mask_list

    def augment_image(self, image, mask):
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
        mask = tf.image.rot90(mask, k=k)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        return image, mask

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_image_paths = [self.image_list[i] for i in indices]
        batch_mask_paths = [self.mask_list[i] for i in indices]
        
        X = np.zeros((self.batch_size, *self.image_size, 1), dtype=np.float32)
        y = np.zeros((self.batch_size, *self.image_size, 1), dtype=np.float32)
        valid_samples = 0
        
        for i, (img_path, mask_path) in enumerate(zip(batch_image_paths, batch_mask_paths)):
            # print(f"Loading image: {img_path}")  # Debug: Check which image is being loaded
            with rasterio.open(img_path) as src:
                img = src.read(1)
                # print(f"Image min: {img.min()}, max: {img.max()}")  # Debug: Check raw values

                img = np.expand_dims(img, axis=-1)

                # Resize using TensorFlow and convert to NumPy array
                img = tf.image.resize(img, self.image_size).numpy()
                # print(f"Resized image min: {img.min()}, max: {img.max()}")  # Debug after resize

                img = normalize_image(img)  # Normalize the image (NumPy array)
                # print(f"Normalized image min: {img.min()}, max: {img.max()}")  # Debug after normalization

            with rasterio.open(mask_path) as src:
                mask = src.read(1)
                mask = np.expand_dims(mask, axis=-1)

                # Resize using TensorFlow and convert to NumPy array
                mask = tf.image.resize(mask, self.image_size).numpy()
                mask = (mask > self.threshold).astype(np.float32)  # Binarize mask

                # Apply small object removal and skeletonization
                mask = remove_small_objects(mask, min_size=self.min_area)
                mask = skeletonize_and_buffer(mask, buffer_size=self.buffer_size)

            # Only use masks that meet the min_area requirement
            if np.sum(mask) < self.min_area:
                continue

            X[valid_samples] = img
            y[valid_samples] = np.expand_dims(mask, axis=-1)
            valid_samples += 1
            if valid_samples >= self.batch_size:
                break
        # np.save(os.path.join(output_dir, f'batch_images_{index}.npy'), X)
        # np.save(os.path.join(output_dir, f'batch_masks_{index}.npy'), y)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def plot_images_masks(images, masks, num=2):
    fig, axs = plt.subplots(num, 2, figsize=(10, 5*num))
    for i in range(num):
        axs[i, 0].imshow(images[i, :, :, 0], cmap='gray')
        axs[i, 0].axis('off')
        axs[i, 0].set_title('Image')
        axs[i, 1].imshow(masks[i, :, :, 0], cmap='gray')
        axs[i, 1].axis('off')
        axs[i, 1].set_title('Mask')
    plt.tight_layout()
    plt.show()

def main(patch_dir, size=512, min_area=500, buffer_size=3, batch_size=8, threshold=75, plot=True):
    """
    Main function to run the data generator script.
    """
    all_files = [f for f in os.listdir(patch_dir) if f.endswith('.tif')]
    train_images = [os.path.join(patch_dir, f) for f in all_files if 'img' in f]
    train_labels = [os.path.join(patch_dir, f.replace('img', 'lab')) for f in train_images if os.path.isfile(os.path.join(patch_dir, f.replace('img', 'lab')))]

    print(f"Total training images: {len(train_images)}")
    print(f"Total training labels: {len(train_labels)}")

    # Check if there are enough files to split
    if len(train_images) == 0 or len(train_labels) == 0:
        raise ValueError("No valid training images or labels found in the provided directory.")

    # change the test_size if the dataset in patch_dir is small
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

    print(f"Training images: {len(train_images)}, Validation images: {len(val_images)}")

    image_size = (size, size)
    train_gen = DataGenerator(train_images, train_labels, batch_size=batch_size, image_size=image_size, shuffle=True, min_area=min_area, buffer_size=buffer_size, threshold=threshold)
    val_gen = DataGenerator(val_images, val_labels, batch_size=batch_size, image_size=image_size, shuffle=False, min_area=min_area, buffer_size=buffer_size, threshold=threshold)

    # Check if the validation generator is empty
    if len(val_gen) == 0:
        raise ValueError("Validation set is empty. Please check your dataset or adjust your split ratio.")

    X_batch, y_batch = train_gen[np.random.randint(len(train_gen))]

    # Use validation set only if it contains data
    if len(val_gen) > 0:
        X_val_batch, y_val_batch = val_gen[np.random.randint(len(val_gen))]
        print("Validation batch shapes:", X_val_batch.shape, y_val_batch.shape)
        print("Max and min in X_val_batch:", X_val_batch.max(), X_val_batch.min())
        print("Max and min in y_val_batch:", y_val_batch.max(), y_val_batch.min())

    print("Training batch shapes:", X_batch.shape, y_batch.shape)
    print("Max and min in X_batch:", X_batch.max(), X_batch.min())
    print("Max and min in y_batch:", y_batch.max(), y_batch.min())

    if plot:
        plot_images_masks(X_batch, y_batch)

# Specify patch directory and call the main function directly
patch_dir = r"C:\Users\14094\trails_tracks_mapper1\patches"
main(patch_dir)

# # Directory to save outputs
# output_dir = r"C:\Users\14094\trails_tracks_mapper1\output"
#
# # Create the output directory if it doesn't exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)