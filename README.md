# DeepFool Adversarial Attack 

This python code is based on the DeepFool adversarial attack implementation by LTS4 [[1]](https://github.com/LTS4/DeepFool), with vast amount of refactoring in the deepfool.py and test_deepfool.py Files.

The algorithm proposed in [[2]](http://arxiv.org/pdf/1511.04599) is an iterative and efficient method to generate adversarial images. The code provided here is a simple implementation of the algorithm in PyTorch.

## What's New in the deepfool.py File?

### Enhanced Device Handling:
- **Streamlined CUDA Handling**: The new implementation uses the `torch.device` method for more streamlined handling of CUDA availability.
- **Automatic Device Assignment**: The image and model are automatically moved to the appropriate device (either GPU or CPU) without the need for explicit checks.

### Image Input Shape:
- **PyTorch Standard**: The updated code expects the image to be of shape CxHxW (channels first), which is more in line with PyTorch's standard conventions.

### Gradient Computation:
- **PyTorch-native Approach**: The new code leverages `torch.autograd.grad` for gradient computation, providing a more direct approach.

### Reduced Dependency on Numpy:
- **Consistency**: The updated implementation minimizes the switching between PyTorch tensors and numpy arrays, making the code more consistent and potentially faster.

### Improved Cloning:
- **Better Compatibility**: Instead of using `copy.deepcopy`, the new code utilizes the `clone` method of a tensor for creating image copies.

### Enhanced Documentation:
- **Clarity**: The first file comes with a more detailed docstring and inline comments, making it easier for users to understand the purpose and functionality of each segment of the code.


## What's New in the test_deepfool.py File?

### **User Interactivity**:
- **Dynamic Image Input**: Users can now input any image filename, making the script more versatile compared to the hardcoded filename in the previous version.

### **Modularity**:
- **Separate Testing Function**: The new script introduces the `test_deepfool` function, allowing for easier reuse and modification of the DeepFool testing process.

### **Image Display**:
- **Side-by-Side Comparison**: The updated script displays both the original and perturbed images side by side for a clearer comparison.

### **Label Mapping**:
- **JSON for Label Mapping**: The updated script uses a JSON file (`imagenet_class_index.json`) for a more structured mapping of class indices to labels, as opposed to the text file in the previous version.

### **Code Structure and Documentation**:
- **Enhanced Documentation**: The first file offers a more detailed docstring and inline comments, aiding users in understanding the code's purpose and functionality.
- **Modern Preprocessing**: The image preprocessing in the new script uses a more modern approach with `transforms.Compose`, streamlining the process.

## Reference
- [1] EPFL LTS4 (https://www.epfl.ch/labs/lts4)
- [2] S. Moosavi-Dezfooli, A. Fawzi, P. Frossard:
*DeepFool: a simple and accurate method to fool deep neural networks*.  In Computer Vision and Pattern Recognition (CVPR ’16), IEEE, 2016.