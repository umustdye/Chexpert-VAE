"""
Author: Heidi Dye
Date: 9/13/2021
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
        #[1, 2320, 2320]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), #8, 1160, 1160
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), #16, 580, 580
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), #32, 290, 290
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), #64, 145, 145
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, stride=2, padding=1), #128, 72, 72
            nn.ReLU(),
            nn.Conv2d(128, 256, 9, stride=9, padding=1), #256, 8, 8
            nn.ReLU(),
            nn.Conv2d(256, 512, 8), #512, 1, 1
            nn.ReLU(),
            Compress()
            )
        
        self.fc_mu = nn.Linear(512, 512)
        self.fc_logvar = nn.Linear(512, 512)
        
        #512, 1, 1
        self.decoder = nn.Sequential(
            Decompress(),
            nn.ConvTranspose2d(512, 256, 7), #256, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=4, padding=1, output_padding=3), #128, 29, 29
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=5, padding=1, output_padding=2), #64, 145, 145
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), #32, 290, 290
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), #16, 580, 580
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1), #8, 1160, 1160
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1), #1, 2320, 2320
            nn.Sigmoid()
            )
     
        
    def generateExample(self, num_examples, device):
        #x must be 128, 1, 1 tensor
        mu = torch.zeros(num_examples, 512).to(device)
        #mu = torch.tensor(np.full((num_examples, 128), -1)).to("cuda")
        logvar = torch.ones(num_examples, 512).to(device)
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
    def forward(self, input):
        #print(input.view(input.size(0), input.size(1), 1, 1).shape)
        return input.view(input.size(0), input.size(1), 1, 1)

def loss_fn(recon_x, x, mu, logvar):
    #use squared error MSELoss
    loss = functional.mse_loss(recon_x, x, reduction="sum")
    #BCELoss = functional.binary_cross_entropy(recon_x, x, reduction="sum")
    KL_Div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loss + KL_Div
    #return BCELoss + KL_Div


#------------------------------------------------------------------------------------------------------------

#open the csv files for the train and valid, respectively, to retrieve individual image paths
csv_filepath = "CheXpert-v1.0/"
train_csv = pd.read_csv(csv_filepath+"train.csv", usecols=["Path"]).values
valid_csv = pd.read_csv(csv_filepath+"valid.csv", usecols=["Path"]).values


batch_size = 16

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
#print("Using device: "+str(device))
#train_dataloader = DataLoader(train_dataset, batch_size=batch_size)



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
    plt.savefig('progress/recon_orig'+str(epoch)+'.png')
    #plt.show(block=True)

#get the data
train_csv = train_csv.squeeze()
valid_csv = valid_csv.squeeze()



#[1, 2828, 2320] -> [1, 2320, 2320]
train_dataset = CHEXPERT_Dataset(csv_file = train_csv, root_dir = csv_filepath, transform=transforms.Resize((2320,2320),interpolation=Image.NEAREST))
valid_dataset = CHEXPERT_Dataset(csv_file = valid_csv, root_dir = csv_filepath, transform=transforms.Resize((2320,2320),interpolation=Image.NEAREST))


def showExample(img):
    #unnormalize
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    #img = img/2 + 0.5
    img = img.numpy()
    plt.title(label="Generated Example")
    plt.savefig('generated_example.png')
    #plt.show(block=True)



#show_image(train_dataset[0].to("cpu"))

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)

print("Number of Images in Valid Dataset: " + str(len(valid_dataset)))    
      
#train the model
model = CHEXPERT_VAE().to(device)


learning_rate = .001
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
epochs = 100



for t in range(epochs):
    #print(f"Epoch {t+1}\n-----------------------------------")
    print("Epoch "+str(t+1)+"\n---------------------------------")
    for idx, (images) in enumerate(valid_dataloader):
        running_loss = 0.0
        images = images.float().to(device)
        recon_x, mu, logvar = model(images)
        loss = loss_fn(recon_x, images, mu, logvar)
        #zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #clean the garbage collector
        gc.collect() 
               
        # print statistics
        running_loss += loss.item()
        if idx % 50 == 49:    # print every 50000 mini-batches
            print('loss: %.3f' %(running_loss / 50))
            running_loss = 0.0
        
          
    if t % 20 == 19:
        #output the original image and the reconstructed image
        showImage(torchvision.utils.make_grid(images.to("cpu")), torchvision.utils.make_grid(recon_x.to("cpu")), t+1)
        #save model progress
        print("Saving model progress...")
        torch.save(model.state_dict(), "CHEXPERT_VAE_MODEL_TEST.pt")
    print("Done!")
    
    
 
print("Saving model...")
torch.save(model.state_dict(), "CHEXPERT_VAE_MODEL_TEST.pt")    


'''
#load the saved model (500 epoches)
print("Loading previously saved model..")
#model = VAE().to(device)

model.load_state_dict(torch.load("CHEXPERT_VAE_MODEL.pt"))
#print(model)
model.eval()
print("Model has been loaded.")
'''


#generate a random example
print("Generating a random example...")
num_examples = 1
#run the example through the decoder
example = model.generateExample(num_examples, device)
showExample(torchvision.utils.make_grid(example.to("cpu")))


#print("ccxd slut machine") 


