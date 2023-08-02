import set_path
import pandas as pd
import numpy as np
import laspy
from upc_analysis import visualization
import multiprocessing

# Paths - point cloud data
base_folder = '../datasets'
dataset_folder = f'{base_folder}/pointcloud/'  # folder with point clouds
pred_folder = f'{base_folder}/predictions/'  # folder with predictions as npz files
prefix = 'filtered_'
prefix_pred = 'pred_'

# Paths - extracted poles
output_file_filter = f'{base_folder}/poles_extracted_filtered.csv'
img_out_folder = f'{base_folder}/images/'

# Load poles data
poles_df = pd.read_csv(output_file_filter)
poles_df.sort_values('tilecode', inplace=True)

# Save png of object x, y and 3d axis
visualization.create_images_for_poles(poles_df, dataset_folder, pred_folder, img_out_folder, prefix, prefix_pred)