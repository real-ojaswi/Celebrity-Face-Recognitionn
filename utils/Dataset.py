import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


class CelebrityDataset(Dataset):
    """It returns image and label if is_test=False and only image if is_test=True.
    csv_file is not required if is_test=False but if is_test=True, a csv_file containing labels is needed."""
    def __init__(self, root_dir, csv_file=None, transform=None, is_test=False):
        self.root_dir = root_dir
        self.images= os.listdir(root_dir)
        if csv_file is not None:
            self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.is_test= is_test
        if (self.is_test==False and csv_file==None):
            raise ValueError('If is_test=False, csv_file is required')



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name= self.images[idx]
        img_full_name = os.path.join(self.root_dir, img_name)
        image = Image.open(img_full_name)
        if self.transform:
                image = self.transform(image)
        if self.is_test==False:
            label = self.get_label_from_csv(self.images[idx])
            return image, label
        else:
            return image
        
    def get_label_from_csv(self, image_filename):
        row = self.data[self.data.iloc[:, 1] == image_filename]
        if row.empty:
            raise ValueError(f"No label found for image '{image_filename}' in the CSV file.")

        # Retrieve label from the fourth column (assuming labels are in the fourth column)
        label = row.iloc[0, 3]
        return label



class CelebrityDatasetTest(Dataset):
    """It is same as CelebrityDatasetTest with only difference being is_test's default value is True and
    it returns img_name as well."""
    def __init__(self, root_dir, csv_file=None, transform=None, is_test=True):
        self.root_dir = root_dir
        self.images= os.listdir(root_dir)
        if csv_file is not None:
            self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.is_test= is_test
        if (self.is_test==False and csv_file==None):
            raise ValueError('If is_test=False, csv_file is required')



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name= self.images[idx]
        img_full_name = os.path.join(self.root_dir, img_name)
        image = Image.open(img_full_name)
        if self.transform:
                image = self.transform(image)
        if self.is_test==False:
            label = self.get_label_from_csv(self.images[idx])
            return image, label, img_name
        else:
            return image, img_name
        
    def get_label_from_csv(self, image_filename):
        row = self.data[self.data.iloc[:, 1] == image_filename]
        if row.empty:
            raise ValueError(f"No label found for image '{image_filename}' in the CSV file.")

        # Retrieve label from the fourth column (assuming labels are in the fourth column)
        label = row.iloc[0, 3]
        return label

if __name__=='__main__':
    print("This file can only be imported!!!")