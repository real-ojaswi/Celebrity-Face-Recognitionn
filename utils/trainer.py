import torch
import torch.nn as nn
import time
import os
from torchvision import models

class Trainer():
    def __init__(self, model, train_loader, val_loader, criterion, device):
        self.model= model.to(device)
        self.train_loader= train_loader
        self.val_loader= val_loader
        self.criterion= criterion
        self.device= device


    def train(self, optimizer, epochs, save_dir, weight= None, start_epoch=0): 
        self.model.train()                  # start_epoch is the epoch index you intend to start from if you've conducted 
        loss_fn= self.criterion             # training previously as well. this will prevent the previously saved 
        train_loader= self.train_loader     # checkpoints from being replaced. check the naming convention for the checkpoints below
        device= self.device
        if weight is not None:
            try:
                self.model.load_state_dict(torch.load(weight))
                print(f'Weight successfullly loaded from {weight}')
            except Exception as e:
                raise ValueError('Provide a valid weight file')
        train_loss_every_epoch=[]
        start_time= time.perf_counter()
        for epoch in range(start_epoch, start_epoch+epochs):
            train_loss= []
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels= images.to(device), labels.to(device)
                labels_pred=self.model(images)
                labels_tensor = labels.clone().detach()
                loss=loss_fn(labels_pred, labels_tensor.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss)

                #to print about 10 logs for every batch_size
                length = len(train_loader)
                tenth_length = round(length / 10)
                if batch_idx % (tenth_length + 1) == 0:
                    current_time= time.perf_counter()
                    elapsed_time= -start_time+current_time
                    print(f'Epoch {epoch:3d}: [{batch_idx*len(images):6d}/{len(train_loader.dataset):6d}]'
                           f'     Elapsed Time: {elapsed_time:5.2f}s   Loss: {torch.mean(torch.FloatTensor(train_loss)):7.3f}')
            assert len(train_loss) == len(train_loader)
            avg_train_loss= torch.mean(torch.FloatTensor(train_loss))
            train_loss_every_epoch.append(avg_train_loss)
            os.makedirs(save_dir, exist_ok=True)
            final_save_dir= os.path.join(save_dir, f'checkpoint_{epoch}.pth')
            torch.save(self.model.state_dict(), final_save_dir)
            self.test(epoch)

        print(f'Training completed for {epoch} epochs!')
        return train_loss

    def test(self, epoch=None, weight= None): #epoch is just for logging purpose
        self.model.eval()
        val_loader= self.val_loader
        loss_fn= self.criterion
        device= self.device
        if weight is not None:
            try:
                self.model.load_state_dict(torch.load(weight))
                print(f'Weight successfullly loaded from {weight}')
            except Exception as e:
                raise ValueError('Provide a valid weight file')        
        
        val_stat={}
        all_accuracy=[]
        all_loss=[]
        all_predictions=[]
        all_labels=[]
        total_num=0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels= images.to(device), labels.to(device)
                labels_pred = self.model(images).float()
                labels_tensor = labels.clone().detach()
                loss=loss_fn(labels_pred, labels_tensor.long())
                accuracy=torch.sum((labels_pred.argmax(dim=1) == labels_tensor).float())
                all_loss.append(loss.item())
                all_accuracy.append(accuracy.item())
                all_predictions.append(labels_pred)
                all_labels.append(labels)
                total_num+=len(images)
                


        val_stat['loss'] = sum(all_loss)/len(all_loss)
        val_stat['accuracy']=sum(all_accuracy)/total_num
        val_stat['prediction']=torch.cat(all_predictions, dim=0)
        val_stat['labels']=torch.cat(all_labels, dim=0)

        if epoch is None:
            epoch='N/A'
        print(f"Test/Validation result at epoch: {epoch:3d}: total sample: {total_num: 6d}, Avg loss: {val_stat['loss']:7.3f}, Acc: {100*val_stat['accuracy']:7.3f}%")
        return val_stat


class TrainerAutoencoder():

    """trainer for autoencoder architecture to get proper feature extractor"""
    def __init__(self, model, train_loader, val_loader, criterion, device):
        self.model= model.to(device)
        self.train_loader= train_loader
        self.val_loader= val_loader
        self.criterion= criterion
        self.device= device


    def train(self, optimizer, epochs, save_dir, weight= None, start_epoch=0): 
        self.model.train()                  # start_epoch is the epoch index you intend to start from if you've conducted 
        loss_fn= self.criterion             # training previously as well. this will prevent the previously saved 
        train_loader= self.train_loader     # checkpoints from being replaced. check the naming convention for the checkpoints below
        device= self.device
        if weight is not None:
            try:
                self.model.load_state_dict(torch.load(weight))
                print(f'Weight successfullly loaded from {weight}')
            except Exception as e:
                raise ValueError('Provide a valid weight file')
        train_loss_every_epoch=[]
        start_time= time.perf_counter()
        for epoch in range(start_epoch, start_epoch+epochs):
            train_loss= []
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels= images.to(device), labels.to(device)
                images_pred=self.model(images)
                # labels_tensor = labels.clone().detach()
                loss=loss_fn(images_pred, images)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss)

                #to print about 10 logs for every batch_size
                length = len(train_loader)
                tenth_length = round(length / 10)
                if batch_idx % (tenth_length + 1) == 0:
                    current_time= time.perf_counter()
                    elapsed_time= -start_time+current_time
                    print(f'Epoch {epoch:3d}: [{batch_idx*len(images):6d}/{len(train_loader.dataset):6d}]'
                           f'     Elapsed Time: {elapsed_time:5.2f}s   Loss: {torch.mean(torch.FloatTensor(train_loss)):7.3f}')
            assert len(train_loss) == len(train_loader)
            avg_train_loss= torch.mean(torch.FloatTensor(train_loss))
            train_loss_every_epoch.append(avg_train_loss)
            final_save_dir= os.path.join(save_dir, f'checkpoint_{epoch}.pth')
            torch.save(self.model.state_dict(), final_save_dir)
            self.test(epoch)

        print(f'Training completed for {epoch} epochs!')
        return train_loss

    def test(self, epoch=None, weight= None): #epoch is just for logging purpose
        self.model.eval()
        val_loader= self.val_loader
        loss_fn= self.criterion
        device= self.device
        if weight is not None:
            try:
                self.model.load_state_dict(torch.load(weight))
                print(f'Weight successfullly loaded from {weight}')
            except Exception as e:
                raise ValueError('Provide a valid weight file')        
        
        val_stat={}
        all_loss=[]
        total_num=0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels= images.to(device), labels.to(device)
                images_pred = self.model(images).float()
                # labels_tensor = labels.clone().detach()
                loss=loss_fn(images_pred, images)

                all_loss.append(loss.item())
                total_num+=len(images)
                


        val_stat['loss'] = sum(all_loss)/len(all_loss)

        if epoch is None:
            epoch='N/A'
        print(f"Test/Validation result at epoch: {epoch:3d}: total sample: {total_num: 6d}, Avg loss: {val_stat['loss']:7.3f}")
        return val_stat
    
class autoencoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone= models.efficientnet_b0()
        self.bottleneck1= nn.Linear(1000, 2352)
        self.convT1= nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.convT2= nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.convT3= nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)
        self.bn1= nn.BatchNorm2d(3)
        self.bn2= nn.BatchNorm2d(3)
        self.bn3= nn.BatchNorm2d(3)
        self.tanh= nn.Tanh()

    def forward(self, x):
        x= self.backbone(x)
        x= nn.functional.relu(self.bottleneck1(x))
        x= x.view(-1, 3, 28, 28)
        x= nn.functional.relu(self.bn1(self.convT1(x)))
        x= nn.functional.relu(self.bn2(self.convT2(x)))
        x= self.tanh(self.bn3(self.convT3(x)))
        return x