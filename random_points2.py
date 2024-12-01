import random
import geopandas as gpd
import rasterio
from shapely.geometry import Point
from pyproj import CRS

def generate_random_points_in_tif(tif_path, num_points, output_shapefile):
    """
    Generate random points within the bounds of a TIFF file and save them as a shapefile.

    Parameters:
    tif_path (str): Path to the input TIFF file.
    num_points (int): Number of random points to generate.
    output_shapefile (str): Path to save the output shapefile with random points.
    """
    # Step 1: Open the TIFF file and extract the bounds and CRS
    with rasterio.open(tif_path) as src:
        bounds = src.bounds  # Get the geographic bounds of the TIFF
        crs = src.crs  # Get the Coordinate Reference System (CRS)

    # Step 2: Generate random points within the bounds
    min_x, min_y, max_x, max_y = bounds.left, bounds.bottom, bounds.right, bounds.top
    points = []
    for _ in range(num_points):
        random_x = random.uniform(min_x, max_x)
        random_y = random.uniform(min_y, max_y)
        points.append(Point(random_x, random_y))

    # Step 3: Create a GeoDataFrame and save it as a shapefile
    gdf = gpd.GeoDataFrame(geometry=points, crs=crs)
    gdf.to_file(output_shapefile)

    print(f"Random points shapefile saved to {output_shapefile}")
    return CRS.from_string(crs.to_wkt())  # Return the CRS as a pyproj CRS


def check_crs_match(tif_crs, shp_crs):
    """
    Check if the CRS of the TIFF and shapefile match.

    Parameters:
    tif_crs: CRS of the TIFF file.
    shp_crs: CRS of the shapefile.

    Returns:
    bool: True if CRS match, False otherwise.
    """
    return tif_crs.equals(shp_crs)


def match_crs(shp_path, target_crs):
    """
    Match the CRS of the shapefile to the target CRS.

    Parameters:
    shp_path (str): Path to the shapefile.
    target_crs: Target CRS to match.
    """
    gdf = gpd.read_file(shp_path)
    if not gdf.crs.equals(target_crs):
        gdf = gdf.to_crs(target_crs)  # Reproject to target CRS
        gdf.to_file(shp_path)  # Save the reprojected shapefile
        print(f"Shapefile CRS matched to the target CRS: {target_crs}")
    else:
        print("Shapefile CRS already matches the target CRS.")


# Example usage:
tif_path = r'C:\Users\14094\trails_tracks_mapper1\droneDTM10cm\img.tif'
num_points = 500  # Specify how many random points you need
output_shapefile = r'C:\Users\14094\trails_tracks_mapper1\shpfile_random2\random_points.shp'

# Generate random points and get the CRS of the TIFF
tif_crs = generate_random_points_in_tif(tif_path, num_points, output_shapefile)

# Check the CRS of the shapefile
shp_crs = gpd.read_file(output_shapefile).crs

# Compare the CRS of the TIFF and shapefile
if check_crs_match(tif_crs, shp_crs):
    print("The CRS of the TIFF and shapefile match.")
else:
    print("The CRS of the TIFF and shapefile do not match.")

# Match the CRS of the shapefile to that of the TIFF
match_crs(output_shapefile, tif_crs)