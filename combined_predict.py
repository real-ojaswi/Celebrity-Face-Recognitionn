from utils.MakePrediction import *
from utils.Dataset import *
from torchvision import datasets,transforms, models
import torch.nn as nn
import torch

if __name__== '__main__':
        
    transform= transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),  # Convert image to RGB if not already in RGB format
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),  # Convert PIL image to Tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize RGB image
    ])

    test_dir= 'data/test_cropped'
    test_dataset= CelebrityDatasetTest(root_dir=test_dir, transform=transform, is_test=True)
    test_dataloader= DataLoader(test_dataset, batch_size=16, shuffle=False)
    # dataloader ends here


    # models to use in ensembling
    model1= models.efficientnet_v2_l()
    model1.fc= nn.Linear(1280,100)
    model2= models.efficientnet_b0()
    model2.fc= nn.Linear(1280,100)
    model3= models.resnet152()
    model3.fc= nn.Linear(2048,100)

    models_list= [model1, model2, model3]

    weight_model1= 'EffNetCheckpoints/checkpoint10.pth'
    weight_model2= 'EffNetCheckpoints/checkpoint12.pth'
    weight_model3= 'resnet_full_checkpoints/checkpoint12.pth'

    weights_list= [weight_model1, weight_model2, weight_model3]

    #predictor instance
    my_predictor= predictor(models_list, weights_list, test_dataloader, 'Predictions/predictions_3_28', 'data/category.csv')
    #make prediction
    my_predictor.get_ensembled_prediction()
