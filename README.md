# DeepFool Adversarial Attack 

This project is a PyTorch implementation of the DeepFool adversarial attack algorithm proposed in the paper: S. Moosavi-Dezfooli, A. Fawzi, P. Frossard: *DeepFool: a simple and accurate method to fool deep neural networks*.  In Computer Vision and Pattern Recognition (CVPR ’16), IEEE, 2016.

## The project consists of these files:

- `deepfool.py`: The main implementation of the DeepFool algorithm.
- `test_deepfool.py`: A script for testing the DeepFool algorithm on a sample image. The ResNet-34 model used for classification is pretrained 
- `resize_and_crop.py`: A script for resizing and cropping images to size of 224x224, as requied by the ResNet-34 model. This size is required for the pretrained model to work properly.

## Image source
The images in this project are taken from the Wikimedia Commons. The original images (before getting resized and cropped) are available at the following links:

- [image1.jpg](https://commons.wikimedia.org/wiki/File:(1218241)_%22Parrot_bird%22_Psittaciformes_-_Amazon,_Brazil.jpg)
- [image2.jpg](https://commons.wikimedia.org/wiki/File:Hot_air_balloon_IMGP0348a.jpg)
- [image3.jpg](https://commons.wikimedia.org/wiki/File:Camera_Zenit_11.jpg)
- [image4.jpg](https://commons.wikimedia.org/wiki/File:Chevy_Pickup_1956.jpg)
