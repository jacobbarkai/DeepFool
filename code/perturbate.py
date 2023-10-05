import torch
import torchvision.transforms as transforms
from torchvision.models import resnet34
from PIL import Image
import os
from deepfool import deepfool

def denormalize(tensor, mean, std):
    """Denormalize a tensor using provided mean and std."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def perturbate_image(image_path, model):
    """Perturbate the image using DeepFool algorithm."""
    
    # Set the device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the image
    image = Image.open(image_path)
    
    # Check if the image size is compatible with ResNet-34
    if image.size != (224, 224):
        print("Image size is not 224x224, which is required by ResNet-34. Please provide an image of the correct size.")
        return None

    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the image
    processed_image = preprocess(image).unsqueeze(0).to(device)
    
    # Apply DeepFool to perturbate the image
    _, _, _, _, perturbed_image = deepfool(processed_image, model)
    
    # Denormalize and clamp the perturbed image
    perturbed_image = denormalize(perturbed_image.squeeze(0), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    # Convert tensor back to PIL Image and return
    return transforms.ToPILImage()(perturbed_image)

if __name__ == "__main__":
    # Load the pretrained ResNet-34 model
    model = resnet34(pretrained=True).eval()
    
    # Get the image path from the user
    image_path = input("Enter the image file path to perturbate: ")
    
    # Perturbate the provided image
    perturbed_image = perturbate_image(image_path, model)
    
    # If perturbation was successful, save the perturbed image with a new name
    if perturbed_image:
        base_name, ext = os.path.splitext(image_path)
        perturbed_image_path = base_name + "_perturbed" + ext
        perturbed_image.save(perturbed_image_path)