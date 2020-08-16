# reference - docs.python.org.library/argparse.html

from linkfile import *

# Promting input from the user
parser = argparse.ArgumentParser()
# Basic usage
parser.add_argument('data_dir',help='Directory')
# Architecture choose
parser.add_argument('--arch')
# Prompting Hyperparameters
parser.add_argument('--learning_rate',type=float,help='Learning rate for network')
parser.add_argument('--hidden_units',type=int,help='Hidden layer for the network')
parser.add_argument('--epochs',type=int,help='Number of epochs')
# For gpu training
parser.add_argument('--gpu')
# setting directory to save the checkpoint
parser.add_argument('--save_dir',help='Directory to save Trained network')
args = parser.parse_args()
data_dir = args.data_dir
# Saving the trained neural network on respective directory
save_dir = args.save_dir
# architexture for the network
arch = args.arch
# learning rate for the network
learning_rate = args.learning_rate
# Number of hidden layer for the respective arch
hidden_units = args.hidden_units
# epochs for the network
epochs = args.epochs
# Image path 
parser.add_argument('image_path')
# Checkpoint where we can retrieve saved model for image identification
parser.add_argument('checkpoint')
# Return top K most likely value
parser.add_argument('--top_k')
# Top categories
parser.add_argument('--category_names')
# For gpu training
device = args.gpu
# Image path
image_path = args.image_path
# checkpoints defining
checkpoint = args.checkpoint
# Top most K likely values
top_k = args.top_k
# categories name
category_names = args.category_names
