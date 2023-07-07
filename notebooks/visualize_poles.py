import set_path
import pandas as pd
import numpy as np
import shapely.geometry as sg
import laspy
from upc_analysis import visualization
import multiprocessing

# Paths - point cloud data
base_folder = '/home/azureuser/cloudfiles/code/blobfuse/pointcloud/UI'
dataset_folder = f'{base_folder}/Point_Cloud_Data/Amsterdam/Amsterdam_Oost/Unprocessed/All/nl-amsv-201001-7415-laz/las_processor_bundled_out/'
pred_folder = f'{base_folder}/Preds/Amsterdam/Amsterdam_Oost/inference_oost_npz/'
prefix = 'filtered_'
prefix_pred = 'pred_'

# Paths - extracted poles
base_folder_output = '/home/azureuser/cloudfiles/code/blobfuse/ovl'
output_file_filter = base_folder_output + "/20230523-200444_poles_extracted_50_1_150_1_filtered.csv"
img_out_folder = f'{base_folder_output}/images/20230614-132510/'

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
        args=(identifier, points[obj_mask], labels[obj_mask], write_path, 'x'))
        p.start() 
        p = multiprocessing.Process(target=visualization.generate_png_single_axis, 
        args=(identifier, points[obj_mask], labels[obj_mask], write_path, 'y'))
        p.start() 
