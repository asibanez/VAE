# Computes metrics
# v1 -> Code reorganized

#%% Imports
import os
import json
import importlib
import torchvision
from argparse import Namespace
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#%% Path definitions
base_path = 'C:/Users/siban/Dropbox/CSAIL/Projects/17_VAE/00_spy_project/00_data/02_runs/00_TEST_DELETE'
model_path = 'C:/Users/siban/Dropbox/CSAIL/Projects/17_VAE/00_spy_project/01_repo/01_model/model_0_AE_v0.py'
original_img_path = 'C:/Users/siban/Dropbox/CSAIL/Projects/17_VAE/00_spy_project/00_data/00_raw/01_subset/imgs/000001.jpg'

#%% Read data
train_results_path = os.path.join(base_path, 'train_results.json')
test_results_path = os.path.join(base_path, 'test_results.json')

with open(train_results_path) as fr:
    train_results = json.load(fr)

with open(test_results_path) as fr:
    test_results = json.load(fr)

pred = test_results['Y_pred']
pred = [x[0] for x in pred]
gr_truth = test_results['Y_gr_truth']
gr_truth = [x[0] for x in gr_truth]

#%% Compute MSE metrics
MSE_returns = mean_squared_error(gr_truth, pred)
MSE_price = mean_squared_error(price_gr_truth, price_pred)

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

#%% Generate images

# Instantiate model
spec=importlib.util.spec_from_file_location("kuk", model_path)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)
args = Namespace(hidden_dim = 16)
model = model_module.VAE_model(args)

#%% Load image
img_original = torchvision.io.read_image(original_img_path)
plt.imshow(img_original.permute(1,2,0))

#%% Generate image
img_original = img_original.unsqueeze(0).float()
img_pred = model(img_original.float())
img_pred = img_pred.squeeze(0)
plt.imshow(img_pred.int().permute(1,2,0))


