# Computes metrics
# v1 -> Code reorganized

#%% Imports
import os
import json
import torch
import importlib
import torchvision
from glob import glob
from argparse import Namespace
import matplotlib.pyplot as plt

#%% Path definitions
original_img_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/17_VAE/00_spy_project/00_data/00_raw/01_subset/imgs/'
model_path = 'C:/Users/siban/Dropbox/CSAIL/Projects/17_VAE/00_spy_project/01_repo/01_model/model_0_AE_v1.py'
results_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/17_VAE/00_spy_project/00_data/02_runs/05_TEST_6_RELU_128d_100ep'
#path_model_weights = os.path.join(results_folder, 'model.pt.9')
path_model_weights = glob(os.path.join(results_folder, '*.pt.*'))[0]

#%% Read data
train_results_path = os.path.join(results_folder, 'train_results.json')

with open(train_results_path) as fr:
    train_results = json.load(fr)

#%% Plot learning curves
plt.plot(train_results['training_loss'], label = 'train')
plt.plot(train_results['validation_loss'], label = 'val')
plt.xlabel('Epochs')
plt.legend(loc = 'upper right')
plt.grid()
plt.show()

#%% Plot validation curve only
plt.plot(train_results['validation_loss'], label = 'val')
plt.xlabel('Epochs')
plt.legend(loc = 'upper right')
plt.grid()
plt.show()

#%% LOAD MODEL
# Instantiate model
args = Namespace(hidden_dim = 128)

# Load model from module
spec=importlib.util.spec_from_file_location("kuk", model_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

# Instantiate model
model = model_module.VAE_model(args)

# Load model weights
model.load_state_dict(torch.load(path_model_weights, map_location = torch.device('cpu'))['model_state_dict'])
model.eval()

#%% IMAGE GENERATION
# Load original image
img_filename = '000012.jpg'
original_img_path = os.path.join(original_img_folder, img_filename)
img_original = torchvision.io.read_image(original_img_path)

# Generate and preprocess image
img_pred = model(img_original.unsqueeze(0) / 255 - 0.5).detach()
img_pred = ((img_pred + 0.5) * 255).squeeze(0).int()

# Plot images
plt.figure()
fig, axes = plt.subplots(1,2)
axes[0].imshow(img_original.permute(1, 2, 0))
axes[1].imshow(img_pred.permute(1, 2, 0))

#%% IMAGE INTERPOLATION

# Load original images
img_filename_1 = '000026.jpg'
img_filename_2 = '000027.jpg'

original_img_path_1 = os.path.join(original_img_folder, img_filename_1)
original_img_path_2 = os.path.join(original_img_folder, img_filename_2)

img_original_1 = torchvision.io.read_image(original_img_path_1)
img_original_2 = torchvision.io.read_image(original_img_path_2)

# Interpolate images
img_interpol = (img_original_1 + img_original_2) * 0.8

# Generate predictions
img_pred_1 = model(img_original_1.unsqueeze(0) / 255 - 0.5).detach()
img_pred_1 = ((img_pred_1 + 0.5) * 255).squeeze(0).int()

img_pred_2 = model(img_original_2.unsqueeze(0) / 255 - 0.5).detach()
img_pred_2 = ((img_pred_2 + 0.5) * 255).squeeze(0).int()

img_pred_interpol = model(img_interpol.unsqueeze(0) / 255 - 0.5).detach()
img_pred_interpol = ((img_pred_interpol + 0.5) * 255).squeeze(0).int()

#%% Plot images
plt.figure()
fig, plot_matrix = plt.subplots(3,2)

plot_matrix[0,0].imshow(img_original_1.permute(1, 2, 0))
plot_matrix[0,0].axis('off')

plot_matrix[0,1].imshow(img_pred_1.permute(1, 2, 0))
plot_matrix[0,1].axis('off')

plot_matrix[1,0].imshow(img_original_2.permute(1, 2, 0))
plot_matrix[1,0].axis('off')

plot_matrix[1,1].imshow(img_pred_2.permute(1, 2, 0))
plot_matrix[1,1].axis('off')

plot_matrix[2,0].imshow(img_interpol.int().permute(1, 2, 0))
plot_matrix[2,0].axis('off')

plot_matrix[2,1].imshow(img_pred_interpol.permute(1, 2, 0))
plot_matrix[2,1].axis('off')

#%%

























