import set_path
import pandas as pd
import numpy as np
import shapely.geometry as sg
import laspy
from upc_analysis import visualization
import multiprocessing

# Paths - point cloud data
base_folder = '/home/azureuser/cloudfiles/code/blobfuse/ovl'
dataset_folder = f'{base_folder}/pointcloud/Unlabeled/Amsterdam/nl-amsd-200923-7415-laz/las_processor_bundled_out/'  # folder with point clouds
pred_folder = f'{base_folder}/predictions/nl-amsd-200923-7415-laz/'  # folder with predictions as npz files
prefix = 'filtered_'
prefix_pred = 'pred_'

# Paths - extracted poles
output_file_filter = base_folder + "/20230707-191830_poles_extracted_zuidoost_filtered.csv"
img_out_folder = f'{base_folder}/images/20230707-191830/'

# Load poles data
poles_df = pd.read_csv(output_file_filter)
poles_df.sort_values('tilecode', inplace=True)

# Save png of object x, y and 3d axis
open_tile = []
for _, obj in poles_df.iterrows():
    # Get object location and top (per pole)
    identifier = obj.identifier
    obj_location = (obj.rd_x, obj.rd_y, obj.z)
    obj_top = (obj.tx, obj.ty, obj.tz)

    if obj.tilecode != open_tile:
        # Get the point cloud data (per tile)
        cloud = laspy.read(f'{dataset_folder}{prefix}{obj.tilecode}.laz')
        points = np.vstack((cloud.x, cloud.y, cloud.z)).T
        npz_file = np.load(pred_folder + prefix_pred + obj.tilecode + '.npz')
        labels = npz_file['label']
        colors = np.vstack((cloud.red, cloud.green, cloud.blue)).T / (2**16 - 1)
        open_tile = obj.tilecode  # tile_code that is currently open

    # Get a mask for the point cloud around the object's location (per pole)
    obj_mask = visualization.get_mask_for_obj(points, obj_location, obj_top[2])
    if sum(obj_mask) > 0:
        # Save the object for all axes
        write_path = img_out_folder + 'object_all_axes' 
        p = multiprocessing.Process(target=visualization.generate_png_all_axes, 
        args=(identifier, points[obj_mask], labels[obj_mask], write_path, 
        colors[obj_mask], np.vstack((obj_location, obj_top)), False))
        p.start()
        # Save the objects per axis
        write_path = img_out_folder + 'object_per_axis'
        p = multiprocessing.Process(target=visualization.generate_png_single_axis, 
        args=(identifier, points[obj_mask], labels[obj_mask], write_path, colors[obj_mask], 'x'))
        p.start()
        p = multiprocessing.Process(target=visualization.generate_png_single_axis, 
        args=(identifier, points[obj_mask], labels[obj_mask], write_path, colors[obj_mask], 'y'))
        p.start()
