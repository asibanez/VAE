# Imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset

#%% DataClass definition
class VAE_dataset(Dataset):
    def __init__(self, dataset):
        self.X = dataset
                                        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        X = self.X[idx]
        
        return X

#%% Model definition
class VAE_model(nn.Module):
            
    def __init__(self, args):
        super(VAE_model, self).__init__()

        # Variables        
        self.h_dim = 16
    
        # Encoder layers
        self.max_pool = nn.MaxPool2d(kernel_size = (2,2))
        self.flatten = nn.Flatten()
        
        self.conv_enc_0a = nn.Conv2d(3, 32, kernel_size = (3,3), padding ='same')
        self.conv_enc_0b = nn.Conv2d(32, 32, kernel_size = (3,3), padding = 'same')        
        
        self.conv_enc_1a = nn.Conv2d(32, 32*2, kernel_size = (3,3), padding = 'same')
        self.conv_enc_1b = nn.Conv2d(32*2, 32*2, kernel_size = (3,3), padding = 'same')
        
        self.conv_enc_2a = nn.Conv2d(32*2, 32*4, kernel_size = (3,3), padding = 'same')
        self.conv_enc_2b = nn.Conv2d(32*4, 32*4, kernel_size = (3,3), padding = 'same')
        
        self.conv_enc_3a = nn.Conv2d(32*4, 32*8, kernel_size = (3,3), padding = 'same')
        self.conv_enc_3b = nn.Conv2d(32*8, 32*8, kernel_size = (3,3), padding = 'same')
        
        self.fc_mean = nn.Linear(36608, self.h_dim)
        self.fc_sigma = nn.Linear(36608, self.h_dim)
        
        # Decoder layers
        self.fc_dec = nn.Linear(self.h_dim, 36608)
        
        self.upsample_2 = nn.Upsample(scale_factor = 2)
        
        self.conv_dec_0a = nn.Conv2d(32*8, 32*8, kernel_size = (3,3), padding ='same')
        self.conv_dec_0b = nn.Conv2d(32*8, 32*8, kernel_size = (3,3), padding = 'same')        
        
        self.conv_dec_1a = nn.Conv2d(32*8, 32*4, kernel_size = (3,3), padding = 'same')
        self.conv_dec_1b = nn.Conv2d(32*4, 32*4, kernel_size = (3,3), padding = 'same')
        
        self.conv_dec_2a = nn.Conv2d(32*4, 32*2, kernel_size = (3,3), padding = 'same')
        self.conv_dec_2b = nn.Conv2d(32*2, 32*2, kernel_size = (3,3), padding = 'same')
        
        self.conv_dec_3a = nn.Conv2d(32*2, 32, kernel_size = (3,3), padding = 'same')
        self.conv_dec_3b = nn.Conv2d(32, 32, kernel_size = (3,3), padding = 'same')
        
        self.upsample_out = nn.Upsample(size = (218,178))
        self.conv_out = nn.Conv2d(32, 3, kernel_size = (1,1), padding = 'same')

    def forward(self, x):
        device = x.get_device()
        if device == -1: device = 'cpu'
		
        # Encode
        x = self.conv_enc_0a(x)                         # batch_size x 32 x 218 x 178
        x = self.conv_enc_0b(x)                         # batch_size x 32 x 218 x 178
        x = self.max_pool(x)                            # batch_size x 32 x 109 x 89

        x = self.conv_enc_1a(x)                         # batch_size x 64 x 109 x 89
        x = self.conv_enc_1b(x)                         # batch_size x 64 x 109 x 89
        x = self.max_pool(x)                            # batch_size x 64 x 54 x 44

        x = self.conv_enc_2a(x)                         # batch_size x 128 x 54 x 44
        x = self.conv_enc_2b(x)                         # batch_size x 128 x 54 x 44
        x = self.max_pool(x)                            # batch_size x 128 x 27 x 22

        x = self.conv_enc_3a(x)                         # batch_size x 256 x 27 x 22
        x = self.conv_enc_3b(x)                         # batch_size x 256 x 27 x 22
        x = self.max_pool(x)                            # batch_size x 256 x 13 x 11 

        x = self.flatten(x)                             # batch_size x (256 x 13 x 11)

        mean = self.fc_mean(x)                          # batch_size x h_dim
        sigma = self.fc_sigma(x)                        # batch_size x h_dim

        # Sample
        std_mean = torch.zeros(mean.size())             # batch_size x h_dim
        std_sigma = torch.ones(sigma.size())            # batch_size x h_dim
        std_sample = torch.normal(std_mean, std_sigma)  # batch_size x h_dim
        std_sample = std_sample.to(device)				# batch_size x h_dim

        # Reparametrize
        reparam_sample = (std_sample + mean)*sigma      # batch_size x h_dim

        # Decode
        x = self.fc_dec(reparam_sample)                 # batch_size x (256 x 13 x 11)
        x = x.reshape(-1, 256, 13, 11)                  # batch_size x 256 x 13 x 11

        x = self.upsample_2(x)                          # batch_size x 256 x 26 x 22
        x = self.conv_dec_0a(x)                         # batch_size x 256 x 26 x 22 
        x = self.conv_dec_0b(x)                         # batch_size x 256 x 26 x 22 

        x = self.upsample_2(x)                          # batch_size x 256 x 52 x 44
        x = self.conv_dec_1a(x)                         # batch_size x 128 x 52 x 44 
        x = self.conv_dec_1b(x)                         # batch_size x 128 x 52 x 44 

        x = self.upsample_2(x)                          # batch_size x 128 x 104 x 88
        x = self.conv_dec_2a(x)                         # batch_size x 64 x 104 x 88 
        x = self.conv_dec_2b(x)                         # batch_size x 64 x 104 x 88 

        x = self.upsample_2(x)                          # batch_size x 64 x 208 x 176
        x = self.conv_dec_3a(x)                         # batch_size x 32 x 208 x 176 
        x = self.conv_dec_3b(x)                         # batch_size x 32 x 208 x 176

        x = self.upsample_out(x)                        # batch_size x 32 x 218 x 178
        x = self.conv_out(x)                            # batch_size x 3 x 218 x 178
        
        return x
