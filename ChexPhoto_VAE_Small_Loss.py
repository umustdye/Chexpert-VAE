"""
Author: Heidi Dye
Date: 8/9/2021
Version: 1.0
Purpose: utilyze the CheXpert dataset, https://stanfordmlgroup.github.io/competitions/chexpert/
in order to train a VAE to recreate and generate new chest xray images
"""


#choose frontal x-rays

#Libraries to import the dataset
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image
import torchvision
from PIL import Image

import torch.nn as nn
import torch.nn.functional as functional

#garbage collector
import gc

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

#interactive mode
plt.ion()


#---------------------------------------------#
#                CHEXPERT VAE                 #
#---------------------------------------------#
class CHEXPERT_VAE(nn.Module):
    def __init__(self):
        super(CHEXPERT_VAE, self).__init__()
        self.gamma = .5
        #[1, 320, 320]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), #8, 160, 160
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), #16, 80, 80
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), #32, 40, 40
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), #64, 20, 20
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), #128, 10, 10
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), #256, 5, 5
            nn.ReLU(),
            nn.Conv2d(256, 512, 5), #512, 1, 1
            nn.ReLU(),
            Compress()
            )
        
        self.fc_mu = nn.Linear(512, 512)
        self.fc_logvar = nn.Linear(512, 512)
        
        #512, 1, 1
        self.decoder = nn.Sequential(
            Decompress(),
            nn.ConvTranspose2d(512, 256, 5), #256, 5, 5
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), #128, 10, 10
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), #64, 20, 20
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), #32, 40, 40
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), #16, 80, 80
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1), #8, 160, 160
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1), #1, 320, 320
            nn.Sigmoid()
            #nn.Tanh()
            #nn.Hardtanh()
            )
     
        
    def generateExample(self, num_examples):
        #x must be 128, 1, 1 tensor
        mu = torch.zeros(num_examples, 512).to("cuda")
        #mu = torch.tensor(np.full((num_examples, 128), -1)).to("cuda")
        logvar = torch.ones(num_examples, 512).to("cuda")
        #logvar = torch.tensor(np.full((num_examples, 128), 1)).to("cuda")
        z = self.sample(mu, logvar)
        
        return self.decoder(z)
        
    
    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        e = torch.randn_like(std)
        z = mu + (std * e)
        return z
    
    def forward(self, x):
        #print(f"Before encoding: {x.shape}")
        x = self.encoder(x)
        #print(f"After encoding: {x.shape}")
        mu = self.fc_mu(x)
        #print(f"Mean (mu) Shape: {mu.shape}")
        logvar = self.fc_logvar(x)
        #print(logvar.shape)
        z = self.sample(mu, logvar)
        #print(f"Z Shape: {z.shape}")
        recon_x = self.decoder(z)
        #print(f"Reconstructed x After Decoding: {recon_x.shape}")
        return recon_x, mu, logvar      
        

class Compress(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Decompress(nn.Module):
    def forward(self, input, size=320*320):
        #print(input.view(input.size(0), input.size(1), 1, 1).shape)
        return input.view(input.size(0), input.size(1), 1, 1)

def loss_fn(recon_x, x, mu, logvar, model):
    #use squared error MSELoss
    loss = functional.mse_loss(recon_x, x, reduction="sum")
    mse = loss / x.shape[0]
    mse = mse.detach()
    model.gamma = .9*model.gamma + .1*mse
    #self.gamma = min(gamma, mse) #larger batch size
    #BCELoss = functional.binary_cross_entropy(recon_x, x, reduction="sum")
    KL_Div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = loss/(2*model.gamma)
    return loss + KL_Div
    #return BCELoss + KL_Divreturn BCELoss + KL_Div


#------------------------------------------------------------------------------------------------------------

#open the csv files for the train and valid, respectively, to retrieve individual image paths
csv_filepath = "CheXpert-v1.0-small/"
train_csv = pd.read_csv(csv_filepath+"train.csv", usecols=["Path"]).values
valid_csv = pd.read_csv(csv_filepath+"valid.csv", usecols=["Path"]).values

#get a 50,000 subset of the training
#train_csv = train_csv[:50000]
#print(len(train_csv))

batch_size = 1
#batch_size = 8
#batch_size = 16
#batch_size = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
#train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
#print(train_dataset[0].shape) [1, 320, 389]

#transform all the images to [1, 320, 320]



#save generated images into a GIF

class CHEXPERT_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.photo_dir = csv_file
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.photo_dir)

    def __getitem__(self, index):
        img_path = self.photo_dir[index]
        deviceType = "cuda" if torch.cuda.is_available() else "cpu"

        #image = torch.tensor(read_image(self.photo_dir[index]),dtype=float, device=deviceType)
        image = Image.open(img_path)
        transformTen = transforms.Compose([transforms.ToTensor()])
        image = transformTen(image)

            
        if self.transform:
            image = self.transform(image)
            
        #return (image, y_label)        
        return image
        

#show the image
def showImage(img, img_recon, epoch):
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    #img = img/2 + 0.5
    img = img.numpy()
    plt.title(label="Original Epoch: #"+str(epoch))
    plt.imshow(np.transpose(img, (1, 2, 0)))
    fig.add_subplot(1, 2, 2)
    #img_recon = img_recon/2 + 0.5
    img_recon = img_recon.numpy()
    plt.title(label="Reconstruction Epoch: #"+str(epoch))
    plt.imshow(np.transpose(img_recon, (1, 2, 0)))
    plt.savefig('progress_small/recon_orig_smaller'+str(epoch)+'.png')
    #plt.show(block=True)

#get the data
train_csv = train_csv.squeeze()
valid_csv = valid_csv.squeeze()
train_dataset = CHEXPERT_Dataset(csv_file = train_csv, root_dir = csv_filepath, transform=transforms.Resize((320,320),interpolation=Image.NEAREST))
valid_dataset = CHEXPERT_Dataset(csv_file = valid_csv, root_dir = csv_filepath, transform=transforms.Resize((320,320),interpolation=Image.NEAREST))


def showExample(img):
    #unnormalize
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    #img = img/2 + 0.5
    img = img.numpy()
    plt.title(label="Generated Example")
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.savefig('smaller_example.png')
    #plt.show(block=True)

def show_image(image):
    #image = Image.fromarray((255*image[i].permute(1, 2, 0)).numpy().astype(np.uint8))
    image = torchvision.utils.make_grid(image)
    #print(image)
    npimg = image.numpy()
    #print(npimg)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


#show_image(train_dataset[0].to("cpu"))

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
    
      
#train the model
model = CHEXPERT_VAE().to(device)


learning_rate = .001
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
epochs = 1



for t in range(epochs):
    print(f"Epoch {t+1}\n-----------------------------------")
    for idx, (images) in enumerate(train_dataloader):
        running_loss = 0.0
        images = images.to(device).float()
        recon_x, mu, logvar = model(images)
        loss = loss_fn(recon_x, images, mu, logvar, model)
        #zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #clean the garbage collector
        gc.collect() 
        
        
        
        # print statistics
        #running_loss += loss.item()
        #if idx % 25000 == 24999:    # print every 50000 mini-batches
            #print(f"Original Image: {images}\n\n")
            #print(f"Reconstructed Image: {recon_x}")
        
            #output the original image and the reconstructed image
            #showImage(torchvision.utils.make_grid(images.to("cpu")), torchvision.utils.make_grid(recon_x.to("cpu")), t+1)
            #print('loss: %.3f' %(running_loss / 25000))
            #running_loss = 0.0
    if t % 25 == 24:
        #output the original image and the reconstructed image
        showImage(torchvision.utils.make_grid(images.to("cpu")), torchvision.utils.make_grid(recon_x.to("cpu")), t+1)
        #save model progress
        #print("Saving model progress...")
        #torch.save(model.state_dict(), "CHEXPERT_VAE_MODEL_TEST.pt")
    running_loss = loss.item()
    print('loss: %.3f' %running_loss)
    print("Done!")

    
''' 
print("Saving model...")
torch.save(model.state_dict(), "CHEXPERT_VAE_MODEL_SMALL_FULL_500EPOCHS.pt")    
'''

'''
#load the saved model (500 epoches)
print("Loading previously saved model..")
#model = VAE().to(device)

model.load_state_dict(torch.load("CHEXPERT_VAE_MODEL_LARGE_1.pt"))
#print(model)
model.eval()
print("Model has been loaded.")
'''

'''
#generate a random example
print("Generating a random example...")
num_examples = 1
#run the example through the decoder
example = model.generateExample(num_examples)
showExample(torchvision.utils.make_grid(example.to("cpu")))
'''

#print("ccxd slut machine") 
