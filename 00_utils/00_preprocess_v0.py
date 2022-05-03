import os
import glob
import random
import datetime
import torch, torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%% Path definition
#img_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/17_VAE/00_spy_project/00_data/00_raw/01_subset/imgs'
#output_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/17_VAE/00_spy_project/00_data/01_preprocessed/'

img_folder = '/home/sibanez/Projects/02_VAE/00_data/00_raw/imgs'
output_folder = '/home/sibanez/Projects/02_VAE/00_data/01_preprocessed'

#%% Global initialization
test_size = 0.2
random.seed = 1234
max_num_files = 100000

#%% Read images
img_list = glob.os.listdir(img_folder)
train_set = []
for item in tqdm(img_list[0:max_num_files]):
    img_path = os.path.join(img_folder, item)
    img = torchvision.io.read_image(img_path).float()
    train_set.append(img)

train_set = torch.stack(train_set)

#%% Show image
print(f'Number of images in dataset = {len(train_set)}')
#image_idx = 15
#_ = plt.imshow(dataset[image_idx,:,:,:].int().permute(1,2,0))

#%% Split datasets
train_set, dev_set = train_test_split(train_set,
                                      test_size = 0.2,
                                      shuffle = False)

#%% Assert dataset sizes
len_dataset = len(train_set) + len(dev_set)
print(f'Size train set = {len_dataset} / {len(train_set) / len_dataset * 100:.2f}%')
print(f'Size dev set = {len_dataset} / {len(dev_set) / len_dataset * 100:.2f}%')

#%% Save output
print(datetime.datetime.now(), 'Saving datasets')
output_train_path = os.path.join(output_folder, 'model_train.pt')
output_dev_path = os.path.join(output_folder, 'model_dev.pt')
torch.save(train_set, output_train_path)
torch.save(dev_set, output_dev_path)
print(datetime.datetime.now(), 'Done')
