import os
from torch import nn
import torch, torchvision
import matplotlib.pyplot as plt

#%% Read image
img_folder = 'C:/Users/siban/Dropbox/CSAIL/Projects/17_VAE/00_spy_project/00_data/01_subset/imgs'
path = os.path.join(img_folder, '000001.jpg') 
img = torchvision.io.read_image(path)

#%% Show image
plt.imshow(img.permute(1,2,0))
img = img.repeat(2,1,1,1)

#%% Define layers

#x = nn.Conv2d(num_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)

h_dim = 16

# ENCODER
max_pool = nn.MaxPool2d(kernel_size = (2,2))
flatten = nn.Flatten()

conv_enc_0a = nn.Conv2d(3, 32, kernel_size = (3,3), padding ='same')
conv_enc_0b = nn.Conv2d(32, 32, kernel_size = (3,3), padding = 'same')        

conv_enc_1a = nn.Conv2d(32, 32*2, kernel_size = (3,3), padding = 'same')
conv_enc_1b = nn.Conv2d(32*2, 32*2, kernel_size = (3,3), padding = 'same')

conv_enc_2a = nn.Conv2d(32*2, 32*4, kernel_size = (3,3), padding = 'same')
conv_enc_2b = nn.Conv2d(32*4, 32*4, kernel_size = (3,3), padding = 'same')

conv_enc_3a = nn.Conv2d(32*4, 32*8, kernel_size = (3,3), padding = 'same')
conv_enc_3b = nn.Conv2d(32*8, 32*8, kernel_size = (3,3), padding = 'same')

fc_mean = nn.Linear(36608, h_dim)
fc_sigma = nn.Linear(36608, h_dim)

#%% DECODER

fc_dec = nn.Linear(h_dim, 36608)

upsample_2 = nn.Upsample(scale_factor = 2)

conv_dec_0a = nn.Conv2d(32*8, 32*8, kernel_size = (3,3), padding ='same')
conv_dec_0b = nn.Conv2d(32*8, 32*8, kernel_size = (3,3), padding = 'same')        

conv_dec_1a = nn.Conv2d(32*8, 32*4, kernel_size = (3,3), padding = 'same')
conv_dec_1b = nn.Conv2d(32*4, 32*4, kernel_size = (3,3), padding = 'same')

conv_dec_2a = nn.Conv2d(32*4, 32*2, kernel_size = (3,3), padding = 'same')
conv_dec_2b = nn.Conv2d(32*2, 32*2, kernel_size = (3,3), padding = 'same')

conv_dec_3a = nn.Conv2d(32*2, 32, kernel_size = (3,3), padding = 'same')
conv_dec_3b = nn.Conv2d(32, 32, kernel_size = (3,3), padding = 'same')

upsample_out = nn.Upsample(size = (218,178))
conv_out = nn.Conv2d(32, 3, kernel_size = (1,1), padding = 'same')


#%% forward
# ENCODE
x = img.float()

x = conv_enc_0a(x)                              # batch_size x 32 x 218 x 178
x = conv_enc_0b(x)                              # batch_size x 32 x 218 x 178
x = max_pool(x)                                 # batch_size x 32 x 109 x 89

x = conv_enc_1a(x)                              # batch_size x 64 x 109 x 89
x = conv_enc_1b(x)                              # batch_size x 64 x 109 x 89
x = max_pool(x)                                 # batch_size x 64 x 54 x 44

x = conv_enc_2a(x)                              # batch_size x 128 x 54 x 44
x = conv_enc_2b(x)                              # batch_size x 128 x 54 x 44
x = max_pool(x)                                 # batch_size x 128 x 27 x 22

x = conv_enc_3a(x)                              # batch_size x 256 x 27 x 22
x = conv_enc_3b(x)                              # batch_size x 256 x 27 x 22
x = max_pool(x)                                 # batch_size x 256 x 13 x 11 

x = flatten(x)                                  # batch_size x (256 x 13 x 11)

#%%
mean = fc_mean(x)                               # batch_size x h_dim
sigma = fc_sigma(x)                             # batch_size x h_dim

# Sample
std_mean = torch.zeros(mean.size())             # batch_size x h_dim
std_sigma = torch.ones(sigma.size())            # batch_size x h_dim
std_sample = torch.normal(std_mean, std_sigma)  # batch_size x h_dim

# Reparametrize
reparam_sample = (std_sample + mean)*sigma      # batch_size x h_dim

#%% DECODE
x = fc_dec(reparam_sample)                      # batch_size x (256 x 13 x 11)
x = x.reshape(-1, 256, 13, 11)                  # batch_size x 256 x 13 x 11

x = upsample_2(x)                               # batch_size x 256 x 26 x 22
x = conv_dec_0a(x)                              # batch_size x 256 x 26 x 22 
x = conv_dec_0b(x)                              # batch_size x 256 x 26 x 22 

x = upsample_2(x)                               # batch_size x 256 x 52 x 44
x = conv_dec_1a(x)                              # batch_size x 128 x 52 x 44 
x = conv_dec_1b(x)                              # batch_size x 128 x 52 x 44 

x = upsample_2(x)                               # batch_size x 128 x 104 x 88
x = conv_dec_2a(x)                              # batch_size x 64 x 104 x 88 
x = conv_dec_2b(x)                              # batch_size x 64 x 104 x 88 

x = upsample_2(x)                               # batch_size x 64 x 208 x 176
x = conv_dec_3a(x)                              # batch_size x 32 x 208 x 176 
x = conv_dec_3b(x)                              # batch_size x 32 x 208 x 176

x = upsample_out(x)                             # batch_size x 32 x 218 x 178
x = conv_out(x)                                 # batch_size x 3 x 218 x 178

#%%

