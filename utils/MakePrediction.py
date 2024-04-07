import os
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets,transforms, models
import torch.nn as nn
import numpy as np

device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class predictor():
    def __init__(self, models, weights, dataloader, output_file=None, mapping_file= None):
        self.models= models
        self.weights= weights
        self.dataloader= dataloader
        self.mapping_file= mapping_file
        self.output_file= output_file

    def get_loaded_model(self, model, weight):
        model.load_state_dict(torch.load(weight))
        print(f'Weight loaded!!!')

    def get_predictions(self, model, dataloader, device, mapping_file=None, output_file=None):
    
        predictions_df = pd.DataFrame(columns=['Id', 'Label'])
        
        model= model.to(device)
        model.eval()
        for i, (image, image_name) in enumerate(dataloader):
            image= image.to(device)
            with torch.no_grad():
                prediction_logits= model(image)
                prediction_logits= prediction_logits.cpu().numpy()
                predictions= np.argmax(prediction_logits, axis=1)
                image_names = [name.split('.')[0] for name in image_name]
                batch_df= pd.DataFrame({'Id': image_names, 'Label': predictions})
                predictions_df= pd.concat([predictions_df, batch_df])

        print('Prediction complete!!')

        if mapping_file is not None:
            mapping_df= pd.read_csv(mapping_file)
            label_to_category = dict(zip(mapping_df['Label'], mapping_df['Category']))
            predictions_df['Category'] = predictions_df['Label'].map(label_to_category)
        # predictions_df= predictions_df.drop(['Label'], axis=1)
        if output_file is not None:
            predictions_df.to_csv(output_file, index=False)
            print(f'Predictions saved to {output_file}')
        predictions_df['Id']= predictions_df['Id'].astype(int)
        predictions_df['Label']= predictions_df['Label'].astype(int)
        predictions_df= predictions_df.reset_index(drop=True)
        predictions_df= predictions_df.sort_values(by='Id')
        return predictions_df
    
    def _combine_predictions(self, predictions, mapping_file, output_file):
        """Gives the combined prediction by providing the mode of prediction. If two values have same frequency,
        it will use the value from the model that comes earlier. Thus predictions should be given in the right order
        of your priority.
        """
        temporary_prediction= pd.DataFrame()
        for i, prediction in enumerate(predictions):
            temporary_prediction[f'Model_{i}']= prediction['Label']

        for index, row in temporary_prediction.iterrows():
            modes = row.mode()
            if len(modes) == 1:
                temporary_prediction.at[index, 'Mode'] = modes.iloc[0]
            else:
                # temporary_prediction.at[index, 'Mode'] = row.iloc[0]
                mode_indices = {}  # Dictionary to store the indices of each mode
                for mode in modes:
                    mode_indices[mode] = np.where(row == mode)[0][0]  # Find the index of the first occurrence of each mode
                sorted_modes = sorted(mode_indices.items(), key=lambda x: x[1])  # Sort modes based on their indices
                earliest_mode = sorted_modes[0][0]  # Select the mode with the earliest index
                temporary_prediction.at[index, 'Mode'] = earliest_mode
            

        final_prediction= predictions[0]  #just as a placeholder
        final_prediction['Label']= temporary_prediction['Mode']
        final_prediction['Label']= final_prediction['Label'].astype(int)

        #mapping to the category name if mapping file is given
        if self.mapping_file is not None:
            mapping_df= pd.read_csv(mapping_file)
            label_to_category = dict(zip(mapping_df['Label'], mapping_df['Category']))
            final_prediction['Category'] = final_prediction['Label'].map(label_to_category)
        # final_prediction= final_prediction.drop(['Label'], axis=1)
        #saving the output file if output_file is given
        if self.output_file is not None:
            final_prediction.to_csv(output_file, index=False)
            print(f'Predictions saved to {output_file}')
        
        
        return final_prediction
    
    def get_ensembled_prediction(self):
        
        for (model, weight) in zip(self.models, self.weights):
            self.get_loaded_model(model, weight)
        
        predictions=[]
        for i, model in enumerate(self.models):
            prediction= self.get_predictions(model, self.dataloader, device)
            predictions.append(prediction)

        self._combine_predictions(predictions, self.mapping_file, self.output_file)

if __name__=='__main__':
    print("This file can only be imported!!!")




        
