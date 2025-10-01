import torch

def gradient(y, x, grad_outputs=None):
    """Compute gradient of y with respect to x using PyTorch autograd."""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, x, grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def fwd_gradient(y, x):
    """Approximate forward-mode gradient using PyTorch autograd."""
    # First compute the gradient of y with respect to x
    dy_dx = gradient(y, x)
    # Directional derivative in the direction of ones_like(dy_dx)
    dummy = torch.ones_like(dy_dx)
    forward_grad = gradient(dy_dx, x, grad_outputs=dummy)
    return forward_grad