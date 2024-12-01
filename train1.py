# coding: utf-8

import sys

import os
from sklearn.model_selection import train_test_split

# Add the parent directory of 'keras_unet' to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from keras_unet.models import custom_unet
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from data_generator import DataGenerator  # Ensure this is the correct import path for your DataGenerator
from metrics import iou, iou_thresholded  # Ensure these are defined in your utils
from scipy.ndimage import label
from skimage.morphology import skeletonize, dilation, square
import wandb  # Import wandb for tracking

wandb.login(key="22f2efd93dd160facdf285e4410cfc1cff2bb6f1")

# Initialize W&B project with configuration parameters
wandb.init(project="trails-tracks-mapper4", config={
    "learning_rate": 0.000003,
    "epochs": 50,
    "batch_size": 8,
    "filters": 32,
    "input_size": 512
})

def iou(y_true, y_pred, smooth=1e-6):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

def iou_thresholded(y_true, y_pred, threshold=0.5, smooth=1e-6):
    y_pred = tf.cast(y_pred > threshold, dtype=tf.float32)
    return iou(y_true, y_pred, smooth)

# Function to load and split data
def load_data(patch_dir, test_size=0.2):
    """
    Load images and labels, then split into training and validation sets.
    """
    # Find all .tif files in the directory
    all_files = [f for f in os.listdir(patch_dir) if f.endswith('.tif')]
    
    # Separate image files and corresponding label files
    train_images = [os.path.join(patch_dir, f) for f in all_files if 'img' in f]
    train_labels = [os.path.join(patch_dir, f.replace('img', 'lab')) for f in train_images if os.path.isfile(os.path.join(patch_dir, f.replace('img', 'lab')))]
    
    # Ensure that both images and labels exist
    if len(train_images) == 0 or len(train_labels) == 0:
        raise ValueError("No valid training images or labels found in the provided directory.")
    
    # Split data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=test_size, random_state=42)
    
    return train_images, val_images, train_labels, val_labels

def train_model(args):
    input_shape = (args['size'], args['size'], 1)
    print(f"Input shape: {input_shape}")

    model = custom_unet(
        input_shape=input_shape,
        filters=args['filters'],
        use_batch_norm=True,
        dropout=0.3,
        num_classes=1,
        output_activation='sigmoid'
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args['learning_rate']), 
                  loss='binary_crossentropy', 
                  metrics=[iou, iou_thresholded])

    earlystopper = EarlyStopping(patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1)

    checkpoint_filepath = os.path.join(args['checkpoint_dir'], f"trails_tracks_model_epoch_{{epoch:02d}}.h5")

    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )

    # Create data generators for training and validation
    train_gen = DataGenerator(
        image_list=args['train_images'],
        mask_list=args['train_labels'],
        batch_size=args['batch_size'],
        image_size=(args['size'], args['size']),
        shuffle=True,
        augment=True,
        min_area=args['min_area'],
        buffer_size=args['buffer_size'],
        threshold=args['threshold']
    )

    val_gen = DataGenerator(
        image_list=args['val_images'],
        mask_list=args['val_labels'],
        batch_size=args['batch_size'],
        image_size=(args['size'], args['size']),
        shuffle=False,
        min_area=args['min_area'],
        buffer_size=args['buffer_size'],
        threshold=args['threshold']
    )

    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args['epochs'],
        steps_per_epoch=len(train_gen) // args['batch_size'],
        validation_steps=len(val_gen) // args['batch_size'],
        callbacks=[earlystopper, reduce_lr, checkpoint_callback]
    )

    return history


import matplotlib.pyplot as plt


def plot_history(history):
    """Plot training and validation loss and metrics."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot loss
    axs[0].plot(history.history['loss'], label='Training Loss')
    axs[0].plot(history.history['val_loss'], label='Validation Loss')
    axs[0].set_title('Loss over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plot IoU metric
    if 'iou' in history.history:
        axs[1].plot(history.history['iou'], label='Training IoU')
        axs[1].plot(history.history['val_iou'], label='Validation IoU')
    if 'iou_thresholded' in history.history:
        axs[1].plot(history.history['iou_thresholded'], label='Training IoU (Thresholded)')
        axs[1].plot(history.history['val_iou_thresholded'], label='Validation IoU (Thresholded)')

    axs[1].set_title('IoU Metric over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('IoU')
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def main(patch_dir, size=512, filters=32, learning_rate=0.000003, batch_size=8, epochs=50, checkpoint_dir=r'C:\Users\14094\trails_tracks_mapper1\CNN_Models_8', min_area=500, buffer_size=3, threshold=75, plot=True):
    """
    Main function to train the model.
    """
    # Load training and validation data
    train_images, val_images, train_labels, val_labels = load_data(patch_dir)

    # Store all arguments in a dictionary
    args = {
        'train_images': train_images,
        'val_images': val_images,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'size': size,
        'filters': filters,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs,
        'checkpoint_dir': checkpoint_dir,
        'min_area': min_area,
        'buffer_size': buffer_size,
        'threshold': threshold,
        'plot': plot
    }

    # # Train the model
    # train_model(args)

    # Train the model and get the history
    history = train_model(args)

    # Plot history if enabled
    if plot:
        plot_history(history)

# Define each parameter individually
patch_dir = r"C:\Users\14094\trails_tracks_mapper1\patches"  # Directory where all data is currently stored
size = 512
filters = 32
learning_rate = 0.000003
batch_size = 2
epochs = 50
checkpoint_dir = r"C:\Users\14094\trails_tracks_mapper1\CNN_Models_9"
min_area = 500
buffer_size = 3
threshold = 75
plot = True

# Run the main function with the defined parameters
main(patch_dir, size, filters, learning_rate, batch_size, epochs, checkpoint_dir, min_area, buffer_size, threshold, plot)
