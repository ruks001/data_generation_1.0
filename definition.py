# parameters for training (train_test.py)
batch_size = 8
num_epochs = 100
model_path = 'pretrained_weights/unet_pretrained.pth'

# For image generation (generate_dataset.py)
images_paths = ['resources/image0.png', 'resources/image1.png', 'resources/image2.png', ]
masks_paths = ['resources/masks0.png', 'resources/masks1.png', 'resources/masks2.png', ]


test_result_path = 'testing_results/'