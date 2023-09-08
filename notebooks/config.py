# Input paths
base_folder = "../datasets"

dataset_folder = f"{base_folder}/pointcloud/"  # folder with point clouds
pred_folder = f"{base_folder}/predictions/"  # folder with predictions as npz files
prefix = "filtered_"
prefix_pred = "pred_"

ahn_data_folder = f"{base_folder}/ahn/"
bgt_building_file = f"{base_folder}/bgt/bgt_buildings_demo.csv"
trees_file = f"{base_folder}/bgt/bgt_trees_demo.gpkg"
tree_area = None

# Output paths
output_file = f"{base_folder}/poles_extracted.csv"
output_file_filter = f"{base_folder}/poles_extracted_filtered.csv"
img_out_folder = f"{base_folder}/images/"

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

# Filter setting for maximum tree distance (m)
MAX_TREE_DIST = 0.6
