import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, jaccard_score, f1_score, precision_score, recall_score, precision_recall_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

def rtk_points_to_raster(points_path, reference_raster, output_raster, nodata_value=-1):
    """
    Convert RTK points (trail/no-trail) to a raster aligned with the reference raster.
    """
    points = gpd.read_file(points_path)

    with rasterio.open(reference_raster) as ref:
        meta = ref.meta
        shape = ref.shape
        transform = ref.transform

    # Rasterize RTK points, assigning 0 (no-trail) and 1 (trail)
    rtk_raster = rasterize(
        [(geom, val) for geom, val in zip(points.geometry, points['Code'])],
        out_shape=shape,
        transform=transform,
        fill=nodata_value,
        dtype='int16'
    )

    # Update metadata to include nodata value
    meta.update(dtype='int16', nodata=nodata_value)

    # Save the raster to disk
    with rasterio.open(output_raster, 'w', **meta) as dst:
        dst.write(rtk_raster, 1)

    print(f"RTK raster saved to: {output_raster}")

def load_tif_as_array(tif_path, nodata_value=None):
    """
    Load a GeoTIFF as a NumPy float array. Replace nodata values with NaN.
    """
    with rasterio.open(tif_path) as src:
        array = src.read(1).astype(float)  # Convert to float to allow NaN

        if nodata_value is not None:
            array[array == nodata_value] = np.nan  # Replace nodata with NaN

    return array

def threshold_probability(prob_array, threshold=10):
    """
    Convert probability predictions to binary (0 or 1) using a threshold.
    """
    return (prob_array >= threshold).astype(np.uint8)

def mask_nodata(gt_array, prob_array):
    """
    Mask out nodata (NaN) values from both ground truth and prediction arrays.
    """
    # Create a mask for valid (non-NaN) values
    valid_mask = ~np.isnan(gt_array)

    # Apply the mask to both ground truth and predictions
    gt_array = gt_array[valid_mask]
    prob_array = prob_array[valid_mask]

    # # Ensure consistent lengths after masking
    # assert len(gt_array) == len(prob_array), "Inconsistent lengths after masking nodata values."
    # Ensure consistent lengths after masking
    assert len(gt_array) == len(prob_array), f"Inconsistent lengths: {len(gt_array)} vs {len(prob_array)}"

    return gt_array.flatten(), prob_array.flatten()



def plot_confusion_matrix(cm):
    """
    Plot confusion matrix as a heatmap using Seaborn.
    """
    labels = ['No-Trail', 'Trail']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()



def plot_precision_recall_curve(gt_array, prob_array):
    """
    Plot the precision-recall curve and calculate AUC.
    """
    precision, recall, _ = precision_recall_curve(gt_array, prob_array)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
def evaluate_performance(gt_array, prob_array):
    """
    Evaluate predictions using various metrics.
    """
    # Mask out NaN values
    gt_array, prob_array = mask_nodata(gt_array, prob_array)

    # Convert probabilities to binary predictions
    pred_array = threshold_probability(prob_array, threshold=10)


    # Calculate confusion matrix to get TP, TN, FP, FN
    tn, fp, fn, tp = confusion_matrix(gt_array, pred_array, labels=[0, 1]).ravel()

    # Compute metrics
    # Compute confusion matrix and metrics
    cm = confusion_matrix(gt_array, pred_array, labels=[0, 1])
    accuracy = accuracy_score(gt_array, pred_array)
    iou = jaccard_score(gt_array, pred_array, average='binary')
    f1 = f1_score(gt_array, pred_array, average='binary')
    precision = precision_score(gt_array, pred_array, average='binary')
    recall = recall_score(gt_array, pred_array, average='binary')

    # Print evaluation results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")

    # Plot confusion matrix
    plot_confusion_matrix(cm)
    plot_precision_recall_curve(gt_array, prob_array)


def main(rtk_points, reference_raster, probability_tif, threshold=50, nodata_value=-1):
    """
    Main workflow: Convert RTK points to raster, threshold predictions, and evaluate.
    """
    rtk_raster_path = reference_raster.replace('.tif', '_rtk.tif')

    # Convert RTK points to raster
    rtk_points_to_raster(rtk_points, reference_raster, rtk_raster_path, nodata_value)

    # Load ground truth and probability prediction arrays
    gt_array = load_tif_as_array(rtk_raster_path, nodata_value)
    prob_array = load_tif_as_array(probability_tif)

    # Threshold probability predictions to binary
    pred_array = threshold_probability(prob_array, threshold)

    # Evaluate predictions
    evaluate_performance(gt_array, prob_array)

# Example usage
rtk_points_path = r'C:\Users\14094\trails_tracks_mapper1\RTK\RTK_labels.shp'
reference_raster_path = r'C:\Users\14094\trails_tracks_mapper1\CNN_Models_11\may8_L1_WAlranch_ndtm_merged_Human_lesstrails_DTM10_512_byCNN_7ep_256.tif'
probability_tif_path = r'C:\Users\14094\trails_tracks_mapper1\CNN_Models_11\may8_L1_WAlranch_ndtm_merged_Human_lesstrails_DTM10_512_byCNN_7ep_256.tif'

# Run the main workflow with a threshold of 50
main(rtk_points_path, reference_raster_path, probability_tif_path, threshold=50)
