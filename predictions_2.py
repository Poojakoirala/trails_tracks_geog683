# coding: utf-8
import tensorflow as tf

# Limiting GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import sys
import os

# Add the parent directory of 'keras_unet' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import rasterio
from rasterio.windows import Window
from keras_unet.models import custom_unet
from metrics import iou, iou_thresholded
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.morphology import skeletonize, dilation, square, remove_small_objects

# Constants
BLOCK_SIZE = 256

# Helper Functions
def adjust_window(window, max_width, max_height):
    col_off, row_off, width, height = window.flatten()
    if col_off + width > max_width:
        width = max_width - col_off
    if row_off + height > max_height:
        height = max_height - row_off
    return Window(col_off, row_off, width, height)

def pad_to_shape(array, target_shape):
    diff_height = target_shape[0] - array.shape[0]
    diff_width = target_shape[1] - array.shape[1]
    padded_array = np.pad(array, ((0, diff_height), (0, diff_width), (0, 0)), 'constant')
    return padded_array

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val == min_val:
        print("Warning: Image has constant pixel values")
        normalized_image = np.zeros_like(image)
    else:
        normalized_image = (image - min_val) / (max_val - min_val)

    return np.nan_to_num(normalized_image, nan=0.0)

def predict_patch_optimized(model, patch, vis=True):
    if patch.max() != patch.min():
        patch_norm = normalize_image(patch)
    else:
        patch_norm = patch

    num_bands = patch.shape[2] if len(patch.shape) == 3 else 1
    patch_norm = patch_norm.reshape(1, patch_norm.shape[0], patch_norm.shape[1], num_bands)

    pred_patch = model.predict(patch_norm)

    if np.isnan(pred_patch).any() or np.isinf(pred_patch).any():
        print("Warning: NaN or Infinity values found in pred_patch")
        pred_patch = np.nan_to_num(pred_patch, nan=0.0, posinf=255.0, neginf=0.0)

    pred_patch = np.clip(pred_patch * 100, 0, 255).squeeze().astype(np.uint8)

    return pred_patch

def post_process_prediction(pred):
    """Applies post-processing to remove small objects and skeletonize."""
    pred_cleaned = remove_small_objects(pred > 50, min_size=100)
    pred_skeleton = skeletonize(pred_cleaned)
    return dilation(pred_skeleton, square(3))

def predict_tif_optimized(model, path, patch_size, overlap_size, vis, model_name):
    stride = patch_size - overlap_size
    model_base_name = os.path.basename(model_name).replace('.h5', '')

    with rasterio.open(path) as src:
        original_height, original_width = src.shape
        pred_accumulator = np.zeros((original_height, original_width), dtype=np.uint8)

        for i in tqdm(range(0, original_height, stride)):
            for j in range(0, original_width, stride):
                window = Window(j, i, patch_size, patch_size)
                window = adjust_window(window, original_width, original_height)
                patch = src.read(window=window)
                patch = np.moveaxis(patch, 0, -1)

                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    patch = pad_to_shape(patch, (patch_size, patch_size))

                pred_patch = predict_patch_optimized(model, patch, vis)
                col_off, row_off, width, height = map(int, window.flatten())
                pred_patch = pred_patch[:height, :width]

                pred_accumulator[row_off:row_off + height, col_off:col_off + width] = np.maximum(
                    pred_accumulator[row_off:row_off + height, col_off:col_off + width], pred_patch)

        output_path = path[:-4] + f'_{model_base_name}_{overlap_size}.tif'
        profile = src.profile.copy()
        profile.update(dtype='uint8', nodata=0, compress='LZW', tiled=True, blockxsize=BLOCK_SIZE, blockysize=BLOCK_SIZE)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(pred_accumulator[np.newaxis, :, :])

    return output_path

def main(input_tif, model_name, overlap_size=256, visualize=False):
    model = load_model(model_name, custom_objects={'iou': iou, 'iou_thresholded': iou_thresholded})

    input_shape = model.layers[0].input_shape
    print("Input shape of the model:", input_shape)

    patch_size = input_shape[0][1]
    print("Image size:", patch_size)

    output_path = predict_tif_optimized(model, input_tif, patch_size, overlap_size, visualize, model_name)
    print(f'Saved final prediction to: {output_path}')

    # Load prediction for post-processing
    with rasterio.open(output_path) as src:
        prediction = src.read(1)  # Read the first band

    print("Applying post-processing...")
    post_processed = post_process_prediction(prediction)

    # Save post-processed result
    post_processed_path = output_path.replace('.tif', '_post_processed.tif')
    with rasterio.open(output_path) as src:
        profile = src.profile
        profile.update(dtype='uint8')

        with rasterio.open(post_processed_path, 'w', **profile) as dst:
            dst.write(post_processed, 1)

    print(f'Saved post-processed prediction to: {post_processed_path}')

# Variables for prediction
input_tif = r'C:\Users\14094\trails_tracks_mapper1\CNN_Models_11\feb23_L1_WAlranch_ndtm_merged.tif'
model_name = r'C:\Users\14094\trails_tracks_mapper1\models\Human_lesstrails_DTM10_512_byCNN_7ep.h5'
overlap_size = 256
visualize = True

# Run the prediction and post-processing
main(input_tif, model_name, overlap_size, visualize)
