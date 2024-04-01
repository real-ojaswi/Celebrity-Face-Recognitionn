import os
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch

async def process_image(image_path, image_dir, save_dir, mtcnn):
    try:
        final_path = os.path.join(image_dir, image_path)
        img = Image.open(final_path).convert("RGB")
        final_path_to_save = os.path.join(save_dir, image_path)
        img_cropped = mtcnn(img, save_path= final_path_to_save)
        print(f"Saved cropped image: {final_path_to_save}")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

async def process_images_batch(image_paths, image_dir, save_dir, mtcnn):
    tasks = []
    for image_path in image_paths:
        task = asyncio.create_task(process_image(image_path, image_dir, save_dir, mtcnn))
        tasks.append(task)
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_dir = 'data/train'
    save_dir = 'data/train_croppedMTCNN2'
    os.makedirs(save_dir, exist_ok=True)

    mtcnn = MTCNN(image_size=224, device= device)

    image_paths = os.listdir(image_dir)
    cropped_images= os.listdir(save_dir)
    image_paths_to_use = [i for i in image_paths if i not in cropped_images]
    asyncio.run(process_images_batch(image_paths_to_use, image_dir, save_dir, mtcnn))
