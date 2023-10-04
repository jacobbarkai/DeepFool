import torch

def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):
    """
    DeepFool adversarial attack.
    
    Parameters:
    - image: Image tensor of size CxHxW.
    - net: Network (input: images, output: values of activation **BEFORE** softmax).
    - num_classes: Number of classes to test against (default = 10).
    - overshoot: Used as a termination criterion to prevent vanishing updates (default = 0.02).
    - max_iter: Maximum number of iterations for deepfool (default = 50).
    
    Returns:
    - Minimal perturbation that fools the classifier, number of iterations, original label, new estimated label, and perturbed image.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device).clone().requires_grad_()
    net = net.to(device).eval()

    original_output = net(image).flatten()
    original_label = original_output.argmax().item()

    perturbation = torch.zeros_like(image)
    for _ in range(max_iter):
        output = net(image + perturbation)
        output[0, original_label].backward(retain_graph=True)
        grad_orig = image.grad.data.clone()
        image.grad.data.zero_()
        
        label = output.argmax().item()
        if label != original_label:
            break

        pert = float('inf')
        for index in range(num_classes):
            if index == original_label:
                continue

            output[0, index].backward(retain_graph=True)
            cur_grad = image.grad.data.clone()
            image.grad.data.zero_()

            w = cur_grad - grad_orig
            f_k = output[0, index] - output[0, original_label]
            pert_k = abs(f_k.item()) / torch.norm(w.flatten())

            if pert_k < pert:
                pert = pert_k
                w_k = w

        r_i = (pert + 1e-4) * w_k / torch.norm(w_k)
        perturbation += r_i

    perturbed_image = image + perturbation
    return perturbation.detach(), _, original_label, label, perturbed_image.detach()