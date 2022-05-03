import os
import PIL
import glob
import tqdm
import torchvision
import matplotlib.pyplot as plt

#%% Read image
img_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/17_VAE/00_spy_project/00_data/00_raw/img_align_celeba/img_align_celeba'
path = os.path.join(img_folder, '000001.jpg') 
img = torchvision.io.read_image(path)

#%% Show image
plt.imshow(img.permute(1,2,0))

#%% Check image sizes
img_path_list = glob.glob(os.path.join(img_folder, '*'))
for path in tqdm.tqdm(img_path_list):
    img = PIL.Image.open(path)
    if img.size != (178, 218):
        print(path, '\n', img.size)
    

