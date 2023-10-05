import torch
import torchvision.transforms as transforms
from torchvision.models import resnet34
from PIL import Image
import matplotlib.pyplot as plt
import json
from deepfool import deepfool

def denormalize(tensor, mean, std):
    """Denormalize a tensor."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def preprocess_image(image, device):
    """Preprocess an image for ResNet-34."""
    # Define the preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0).to(device)

def perturbate_image(image_path, model, device):
    """Perturbate the image using DeepFool algorithm."""
    # Load the image
    image = Image.open(image_path)
    
    # Check if the image size is compatible with ResNet-34
    if image.size != (224, 224):
        print("Image size is not 224x224, which is required by ResNet-34. Please provide an image of the correct size.")
        return None

    # Preprocess the image
    processed_image = preprocess_image(image, device)
    
    # Apply DeepFool to perturbate the image
    _, _, _, _, perturbed_image = deepfool(processed_image, model)
    
    # Denormalize and clamp the perturbed image
    perturbed_image = denormalize(perturbed_image.squeeze(0), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    # Convert tensor back to PIL Image and return
    return transforms.ToPILImage()(perturbed_image)

def display_images(original, perturbed, original_label, perturbed_label):
    """Display the original and perturbed images side by side."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original)
    ax[0].set_title(f"{original_label} (Original)")
    ax[0].axis('off')
    ax[1].imshow(perturbed)
    ax[1].set_title(f"{perturbed_label} (Perturbed)")
    ax[1].axis('off')
    plt.show()

def get_label(tensor, model, class_labels):
    """Get the label of an image tensor."""
    with torch.no_grad():
        outputs = model(tensor)
        _, label_idx = outputs.max(1)
    return class_labels[label_idx.item()]

if __name__ == "__main__":
    # Set the device (CUDA if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the pretrained ResNet-34 model
    model = resnet34(pretrained=True).eval().to(device)
    
    # Load ImageNet class labels
    with open('imagenet_class_index.json', 'r') as f:
        class_idx = json.load(f)
        class_labels = [item[1] for item in class_idx.values()]

    # Get the image path from the user
    image_path = input("Enter the image file path: ")
    
    # Perturbate the image
    perturbed_image = perturbate_image(image_path, model, device)
    
    # If perturbation was successful, display the original and perturbed images side by side
    if perturbed_image:
        original_image = Image.open(image_path)
        original_label = get_label(preprocess_image(original_image, device), model, class_labels)
        perturbed_label = get_label(preprocess_image(perturbed_image, device), model, class_labels)
        display_images(original_image, perturbed_image, original_label, perturbed_label)
