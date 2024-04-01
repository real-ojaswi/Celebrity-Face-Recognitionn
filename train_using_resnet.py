import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision import models
from torchvision import datasets,transforms
from torch.utils.data import random_split
import torch.optim as optim
import torch.nn as nn
from utils.Dataset import *
from utils.trainer import *

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###################################################################################################################
###########################inputs needed for the execution of file#################################################
root_dir = 'data/train_croppedMTCNN2' # directory with images
csv_file = 'data/train.csv'     # csv file containing labels for images
validation_file= 'data/validation_file.pth'  #file containing the list of indices for the images to be considered in 
                                             # validation set
to_save_dir= 'ResnetCheckpoints_data2' #  directory where checkpoints will be saved
to_use_weight= None # (OPTIONAL) the weight to be loaded

epochs= 20 #the number of epochs to train for 
learning_rate= 0.0001 # learning rate for training
start_epoch= 0   # (OPTIONAL) the epoch to start from (if you're continuing from the last session, use the number 
                  # after your last saved checkpoint)


###################################################################################################################
######################################### FOR PREPARING DATA #####################################################
#file containing the list of indices which will be considered while building validation_dataset
validation_files= torch.load(validation_file)  #this has the indices of the files that are in validation_set

transform= transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),  # Convert image to RGB if not already in RGB format
    transforms.Resize((224, 224)),  # Resize the image to 229x229
    transforms.ToTensor(),  # Convert PIL image to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize RGB image
])   

dataset = CelebrityDataset(root_dir=root_dir, csv_file=csv_file, transform=transform)

val_indices = [i for i, filename in enumerate(dataset.images) if filename in validation_files]
train_indices = [i for i in range(len(dataset)) if i not in val_indices]

training_dataset= Subset(dataset, train_indices)
validation_dataset= Subset(dataset, val_indices)

# Create dataloaders for training and validation sets
train_dataloader = DataLoader(training_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=False)


###################################################################################################################
######################################### MODEL AND TRAINING #####################################################
resnet= models.resnet152(pretrained=True)
resnet.fc= nn.Linear(2048, 100)
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(resnet.parameters(), lr=learning_rate)
my_trainer= Trainer(resnet, train_dataloader, val_dataloader, criterion, device)
my_trainer.train(optimizer, epochs, to_save_dir, to_use_weight, start_epoch= start_epoch)