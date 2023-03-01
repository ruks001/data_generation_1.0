# data_generation

### After clonning the github repository, follow the steps below by simply executing the command given at each step.

# I. PREREQUISITE
### 1. Create virtual environment.
```
python3 -m venv datagen
```

### 2.Activate virtual environment
```
source datagen/bin/activate
```
### 3.Install required packages
```
pip install -r requirements.txt
```
##### for PyTorch
```
pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

# II. MAIN TASK

### 1.Generate and store images. (The images are stored in 'image_data' folder and masks are stored in 'mask_data' folder.)
The images in the "resources" folder is in RGBA format. If you want to add more images to generate data, they must be in RGBA format.
```
python generate_dataset.py
```

### 2.Train and test data. (Training and validation loss and dsc graphs are saved in 'graphs' folder. Pretrained weight is aved in "pretrained_werights" folder.)
```
python train_test.py
```

### 3.Use model to segment in test data. (create folder "testing_images" in the same project folder and store images to be segmented inside.)
The model is trained and tested in the RGB format. Make sure the images in the "testing_images" are in the RGB format.
```
python test_file.py
```

# Graphs showing progress in training.
![DSC_plot.png](graphs%2FDSC_plot.png)