import os
import cv2
import numpy as np
import torch
from definition import *
from Unet_model import UNet


os.makedirs('testing_images', exist_ok=True)

if not os.path.isfile(model_path):
    print(f' There is no pretrained weight in the location: {model_path}')
    exit()

os.makedirs(test_result_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet()
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()


for root, dir, files in os.walk('testing_images/'):
    if len(files) < 1:
        print(" There are no images in 'testing_images' folder")
        break
    print(f'Predicting and saving images in folder: {test_result_path}..................')
    with torch.no_grad():
        for file in files:
            image = cv2.imread(root + file)
            height, width, channel = image.shape
            image = cv2.resize(image, (400, 400), cv2.INTER_LINEAR)
            image = np.moveaxis(image, -1, 0)
            image = np.expand_dims(image, axis=0)
            image_tensor = torch.from_numpy(image).float().to(device)

            output = model(image_tensor).cpu().numpy()
            output = output[0, 0, :, :] * 255
            output = cv2.resize(output, (width, height), cv2.INTER_LINEAR)
            cv2.imwrite(test_result_path + file, output)