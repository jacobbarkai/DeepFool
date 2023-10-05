from PIL import Image
import os

def resize_and_crop(img_path):
    """Resize and crop an image to 224x224 pixels."""
    
    # Open the image
    img = Image.open(img_path)
    
    # Get the dimensions of the image
    width, height = img.size

    # Determine the shortest side of the image
    shortest_side = min(width, height)

    # Calculate the scaling factor to make the shortest side 224 pixels
    scale_factor = 224 / shortest_side

    # Resize the image using the scaling factor
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Calculate the coordinates for a centered 224x224 crop
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = (new_width + 224) / 2
    bottom = (new_height + 224) / 2

    # Crop the image to 224x224 pixels
    img = img.crop((left, top, right, bottom))

    # Save the cropped image with a new name
    base_name, ext = os.path.splitext(img_path)
    new_name = base_name + "_cropped" + ext
    img.save(new_name)
    
    print(f"Saved the cropped image as {new_name}")

if __name__ == "__main__":
    # Get the image path from the user
    img_path = input("Enter the image file path to resize and crop: ")
    
    # Resize and crop the provided image
    resize_and_crop(img_path)