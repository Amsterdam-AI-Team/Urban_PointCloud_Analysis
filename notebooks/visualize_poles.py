"""
In order to verify the results and spot potential issues, we visualize all individual objects. 
The visualization renders the object from two directions (X and Y axis), along with a 3D projection.
"""

import config as cf  # use config or config_azure
import pandas as pd
import set_path

from upc_analysis import visualization

# Load poles data
poles_df = pd.read_csv(cf.output_file_filter)
poles_df.sort_values("tilecode", inplace=True)

# Save image of object x, y and 3d axis
visualization.create_images_for_poles(
    poles_df, cf.dataset_folder, cf.pred_folder, cf.img_out_folder, cf.prefix, cf.prefix_pred
)