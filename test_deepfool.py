import torch
import torchvision.transforms as transforms
from torchvision.models import resnet34
from PIL import Image
import matplotlib.pyplot as plt
import json
from deepfool import deepfool
import numpy as np

def denormalize(tensor, mean, std):
    for i, (m, s) in enumerate(zip(mean, std)):
        tensor[i] = tensor[i] * s + m
    return tensor

def test_deepfool(image_path, model, class_labels):
    # Load and preprocess the image
    image = Image.open(image_path)
    processed_image = preprocess(image).unsqueeze(0).to(device)
    
    # Run DeepFool on the image
    _, _, original_label, perturbed_label, perturbed_image = deepfool(processed_image, model)
    
    # Post-process the perturbed image
    perturbed_image = post_process(perturbed_image.squeeze(0))
    
    # Display images
    display_images(image, perturbed_image, class_labels[original_label], class_labels[perturbed_label])

def post_process(tensor_image):
    if isinstance(tensor_image, np.ndarray):
        tensor_image = torch.from_numpy(tensor_image).float()

    tensor_image = denormalize(tensor_image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tensor_image = tensor_image.clamp(0, 1).permute(1, 2, 0)
    return Image.fromarray((tensor_image.numpy() * 255).astype('uint8'))


def display_images(original, perturbed, original_label, perturbed_label):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(display_preprocess(original))
    ax[0].set_title(f"{original_label} (Original)")
    ax[0].axis('off')
    ax[1].imshow(perturbed)
    ax[1].set_title(f"{perturbed_label} (Perturbed)")
    ax[1].axis('off')
    plt.show()

if __name__ == "__main__":
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet34(pretrained=True).eval().to(device)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    display_preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    
    with open('imagenet_class_index.json', 'r') as f:
        class_idx = json.load(f)
        class_labels = [item[1] for item in class_idx.values()]

    # Get filename from user and test DeepFool
    filename = input("Enter the image filename: ")
    test_deepfool(f"images/{filename}", model, class_labels)