# Number of classes
c = 3 # NOTE : This number includes the ignore label!

# Multiplier for the number of feature maps in the decompressing part
N = 2

# Ignore label
ig_lbl = 0

# Prefix for all layer names
layer_prefix = ""

# Number of images in the test set
testset_size = 800

# Divides the number of feature maps by this number for each layer
feat_div  = 2

# Learn rates for multinet
lw = 1   # Weights
lb = 2   # Biases

# Batch size
batch_size = 3;

# Output directory for the generated prototxt's
output_dir = '../gen_files/'
# Name prefix for the generated prototxt's
file_prefix = 'train'

# Paths to training set
images_db_path = '/home/vertensj/training_data/blender_train_700/images_lmdb0'
labels_db_path = '/home/vertensj/training_data/blender_train_700/labels_lmdb'

#Path to mean file
mean_file = '/home/vertensj/training_data/blender_train_700/mean.binaryproto'

# Paths to test set
test_images_db_path = '/home/vertensj/training_data/blender_test_700/images_lmdb0'
test_labels_db_path = '/home/vertensj/training_data/blender_test_700/labels_lmdb'

# At this iteration a snapshot is created and the network is tested
test_at_iter = 2000