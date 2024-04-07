from utils.MakePrediction import *
from utils.Dataset import *
from torchvision import datasets,transforms, models
import torch.nn as nn
import torch
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # See issue #152
# CUDA_VISIBLE_DEVICES = 2

if __name__== '__main__': 
    transform_224= transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),  # Convert image to RGB if not already in RGB format
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),  # Convert PIL image to Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize RGB image
    ])
    # # for 299x299 trained models
    # transform_299= transforms.Compose([   
    #     transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),  # Convert image to RGB if not already in RGB format
    #     transforms.Resize((299, 299)),  # Resize the image to 299x299
    #     transforms.ToTensor(),  # Convert PIL image to Tensor
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize RGB image
    # ])

    test_dir= 'data/test_croppedMTCNN2'
    test_dataset_224= CelebrityDatasetTest(root_dir=test_dir, transform=transform_224, is_test=True)
    test_dataloader_224= DataLoader(test_dataset_224, batch_size=16, shuffle=False)

    # test_dataset_299= CelebrityDatasetTest(root_dir=test_dir, transform=transform_299, is_test=True)
    # test_dataloader299= DataLoader(test_dataset_299, batch_size=16, shuffle=False)
    # dataloader ends here


    # models to use in ensembling
    model1= models.regnet_y_128gf()
    model1.fc= nn.Linear(7392,100)
    model2= models.regnet_y_32gf()
    model2.fc= nn.Linear(3712,100)
    model3= models.efficientnet_v2_l()
    model3.fc= nn.Linear(1280,100)
    model4= models.efficientnet_b0()
    model4.fc= nn.Linear(1280,100)
    model5= models.resnet152()
    model5.fc= nn.Linear(2048,100)

    models_list= [model1, model2, model3, model4, model5]
    # models_list= [model1, model2, model4]
    weight_model1= 'RegnetCheckpoints_data2/checkpoint_1.pth' #update your weights
    weight_model2= 'RegnetCheckpoints2_data2/checkpoint_1.pth'
    weight_model3= 'EfficientNetCheckpointsv2_data2/checkpoint_3.pth'  #update your weights
    weight_model4= 'EfficientNetCheckpoints_data2/checkpoint_3.pth'    #update your weights
    weight_model5= 'ResnetCheckpoints_data2/checkpoint_5.pth'   #update your weights

    weights_list= [weight_model1, weight_model2, weight_model3, weight_model4, weight_model5]
    # weights_list= [weight_model1, weight_model2, weight_model4]

    #predictor instance
    my_predictor= predictor(models_list, weights_list, test_dataloader_224, 'Predictions/predictions_4_3', 'data/category.csv')
    #make prediction
    my_predictor.get_ensembled_prediction()
