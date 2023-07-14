import set_path
import pandas as pd
import numpy as np
import shapely.geometry as sg
import laspy
from upc_analysis import visualization
import multiprocessing

# Select location
#my_location = '201001' # Oost
#my_location = '200923' # Zuidoost
my_location = '200921' # Centrum
#my_location = '200920' # Nieuw-West
#my_location = '200918' # Noord
#my_location = '200903' # Zuid
#my_location = '200824' # West
#my_location = '200823' # Haven

# Paths - point cloud data
base_folder = '/home/azureuser/cloudfiles/code/blobfuse/ovl'
dataset_folder = f'{base_folder}/pointcloud/Unlabeled/Amsterdam/nl-amsd-{my_location}-7415-laz/las_processor_bundled_out/'  # point clouds
pred_folder = f'{base_folder}/predictions/nl-amsd-{my_location}-7415-laz/'  # predictions as npz files
prefix = 'filtered_'
prefix_pred = 'pred_'

# Paths - extracted poles
output_file_filter = f'{base_folder}/poles_extracted_{my_location}_filtered.csv'
img_out_folder = f'{base_folder}/images/{my_location}/'

# Load poles data
poles_df = pd.read_csv(output_file_filter)
poles_df.sort_values('tilecode', inplace=True)

# Save png of object x, y and 3d axis
visualization.create_images_for_poles(poles_df, dataset_folder, pred_folder, img_out_folder, prefix, prefix_pred)