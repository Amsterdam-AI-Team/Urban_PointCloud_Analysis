# Select location
my_location = '200923' # Zuidoost
#my_location = '200921' # Centrum
#my_location = '200920' # Nieuw-West
#my_location = '200918' # Noord
#my_location = '200903' # Zuid
#my_location = '200824' # West
#my_location = '200823' # Westpoort
#my_location = '201001' # Oost
tree_area = 'Zuidoost'

# Input paths
base_folder = '/home/azureuser/cloudfiles/code/blobfuse/ovl'

dataset_folder = f'{base_folder}/pointcloud/Unlabeled/Amsterdam/nl-amsd-{my_location}-7415-laz/las_processor_bundled_out/'  # folder with point clouds
pred_folder = f'{base_folder}/predictions/nl-amsd-{my_location}-7415-laz/'  # folder with predictions as npz files
prefix = 'filtered_'
prefix_pred = 'pred_'

ahn_data_folder = f'{base_folder}/ahn/Amsterdam/ahn4_npz/'
bgt_building_file = f'{base_folder}/bgt/bgt_buildings_oost.csv'
trees_file = f'{base_folder}/trees/obj_vgo_boom_view.gpkg'

# Output paths
output_file = f'{base_folder}/poles_extracted_{my_location}.csv'
output_file_filter = f'{base_folder}/poles_extracted_{my_location}_filtered.csv'
img_out_folder = f'{base_folder}/images/{my_location}/'

# Define the class we are interested in
target_label = 60

# Labels used for ground points ('Road' and 'Other ground')
ground_labels = [1, 9]

# Settings for noise filtering
EPS_N = 0.1
MIN_SAMPLES_N = 50

# Settings for clustering
EPS = 0.1
MIN_SAMPLES = 150

# Filter settings for minimum and maximum height (m)
MIN_HEIGHT = 1.8
MAX_HEIGHT = 16