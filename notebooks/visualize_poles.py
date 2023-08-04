import set_path
import pandas as pd
import numpy as np
import laspy
from upc_analysis import visualization
import multiprocessing
import config as cf

# Load poles data
poles_df = pd.read_csv(cf.output_file_filter)
poles_df.sort_values('tilecode', inplace=True)

# Save png of object x, y and 3d axis
visualization.create_images_for_poles(poles_df, cf.dataset_folder, cf.pred_folder, cf.img_out_folder, cf.prefix, cf.prefix_pred)