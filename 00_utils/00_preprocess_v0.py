import os
import glob
import random
import torch, torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#%% Path definition
img_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/17_VAE/00_spy_project/00_data/00_raw/01_subset/imgs'
output_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/17_VAE/00_spy_project/00_data/01_preprocessed/'

#%% Global initialization
test_size = 0.2
random.seed = 1234

#%% Read images
img_list = glob.os.listdir(img_folder)
aux = []
for item in tqdm(img_list):
    img_path = os.path.join(img_folder, item)
    img = torchvision.io.read_image(img_path).float()
    aux.append(img)

dataset = torch.stack(aux)

#%% Show image
print(f'Number of images in dataset = {len(dataset)}')
image_idx = 15
_ = plt.imshow(dataset[image_idx,:,:,:].int().permute(1,2,0))

#%% Split datasets
train_set, dev_set = train_test_split(dataset,
                                      test_size = 0.2,
                                      shuffle = False)

#%% Assert dataset sizes
print(f'Size train set = {len(train_set)} / {len(train_set)/len(dataset)*100:.2f}%')
print(f'Size dev set = {len(dev_set)} / {len(dev_set)/len(dataset)*100:.2f}%')

#%% Save output
output_train_path = os.path.join(output_folder, 'model_train.pt')
output_dev_path = os.path.join(output_folder, 'model_dev.pt')
torch.save(train_set, output_train_path)
torch.save(dev_set, output_dev_path)
print('Done')