import set_path
import pandas as pd
import numpy as np
import shapely.geometry as sg
import laspy
from upc_analysis import visualization
import multiprocessing


# Paths - point cloud data
base_folder = '/home/azureuser/cloudfiles/code/blobfuse/ovl'
dataset_folder = f'{base_folder}/pointcloud/Unlabeled/Amsterdam/nl-amsd-200923-7415-laz/las_processor_bundled_out/'  # point clouds
pred_folder = f'{base_folder}/predictions/nl-amsd-200923-7415-laz/'  # predictions as npz files
prefix = 'filtered_'
prefix_pred = 'pred_'

# Paths - extracted poles
output_file_filter = base_folder + "/20230707-191830_poles_extracted_zuidoost_filtered.csv"
img_out_folder = f'{base_folder}/images/20230707-191830/'

# Load poles data
poles_df = pd.read_csv(output_file_filter)
poles_df.sort_values('tilecode', inplace=True)

# Save png of object x, y and 3d axis
visualization.create_images_for_poles(poles_df, dataset_folder, pred_folder, img_out_folder, prefix, prefix_pred)