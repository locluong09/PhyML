from phyml.base import PIMLModule
from phyml.utils import gradient

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


# Define a simple neural network for the 1D Darcy flow problem
class DarcyFlowNet(nn.Module):
    def __init__(self, layers):
        """
        Neural network for the 1D Darcy flow problem.
        
        Args:
            layers: List of layer sizes, e.g., [1, 20, 1]
        """
        super(DarcyFlowNet, self).__init__()
        
        modules = []
        for i in range(len(layers)-2):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        
        self.model = nn.Sequential(*modules)
        
        # Initialize weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.model(x)


# PDE Residual for 1D Darcy's Flow
def pde_residual_1d_darcy(spatial, time, model):
    """
    Compute the PDE residual for Darcy's flow (∇²h = 0).
    
    Args:
        spatial: List containing a tensor of spatial coordinates [x]
        time: None (not used in this steady-state problem)
        model: PIMLModule instance
        
    Returns:
        Loss based on the PDE residual
    """
    x = spatial[0]
    
    # Forward pass to get predictions
    h_pred = model.forward(spatial)
    
    # Compute h_xx (second derivative) using automatic differentiation
    h_x = gradient(h_pred, x)
    h_xx = gradient(h_x, x)
    
    # Darcy's equation in 1D: h_xx = 0
    pde_residual = h_xx
    
    # Return MSE of the residual
    return torch.mean(pde_residual**2)


# Example usage with Darcy flow problem
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(1234)
    np.random.seed(1234)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Domain setup
    xmin = 0.0  # Left boundary
    xmax = 1.0  # Right boundary
    
    # Boundary conditions
    lb_head = 1.0  # Head at left boundary
    rb_head = 0.9  # Head at right boundary
    
    # Number of collocation points
    n_colloc = 100
    
    # Generate random collocation points
    x_colloc = torch.FloatTensor(np.random.uniform(xmin, xmax, (n_colloc, 1))).to(device)
    x_colloc.requires_grad_(True)
    
    # Plot the collocation points
    plt.figure(figsize=(5, 4), dpi=150)
    plt.scatter(x_colloc.detach().cpu().numpy(), np.zeros_like(x_colloc.detach().cpu().numpy()), 
                color='black', label='Collocation Points', s=5)
    plt.xlabel('$x$')
    plt.legend()
    plt.ylabel('Random Points (Plotted on $y=0$)')
    plt.title('Random Collocation Points in 1D Domain')
    plt.savefig('collocation_points.png')
    plt.close()
    
    # Neural network architecture
    n_nodes = 20
    layers = [1, n_nodes, 1]
    
    # Create the neural network
    net = DarcyFlowNet(layers).to(device)
    
    # Hyperparameters
    lr = 1e-3
    epochs = 10000
    
    # Define the PIML module
    piml = PIMLModule(
        net=net,
        pde_loss=pde_residual_1d_darcy,
        optimizer=optim.Adam,
        lr=lr,
        loss_fn='mse'
    ).to(device)
    
    # Define boundary condition points
    x_left = torch.tensor([[xmin]], dtype=torch.float32).to(device)
    h_left = torch.tensor([[lb_head]], dtype=torch.float32).to(device)
    
    x_right = torch.tensor([[xmax]], dtype=torch.float32).to(device)
    h_right = torch.tensor([[rb_head]], dtype=torch.float32).to(device)
    
    # Combine boundary points for boundary loss
    x_boundary = torch.cat([x_left, x_right], dim=0)
    h_boundary = torch.cat([h_left, h_right], dim=0)
    
    # Create batch for training with PDE residual and boundary conditions
    def create_batch():
        batch = {
            "pde_loss": (x_colloc,),  # PDE residual
            "boundary_loss": (x_boundary, h_boundary),  # Both boundaries combined
        }
        return batch
    
    # Training loop
    losses = []
    min_loss = float('inf')
    best_params = None
    best_iteration = 0
    
    print("Starting training...")
    
    for i in range(epochs + 1):
        batch = create_batch()
        loss = piml.train_step(batch)
        losses.append(loss)
        
        if loss < min_loss:
            min_loss = loss
            best_params = {name: param.clone() for name, param in piml.net.named_parameters()}
            best_iteration = i
            
        if i % 100 == 0:
            print(f"Epoch {i}: Loss = {loss:.4e}")
            
            # Check boundary conditions
            with torch.no_grad():
                left_val = piml.predict(x_left).item()
                right_val = piml.predict(x_right).item()
                print(f"  Left BC: {left_val:.4f} (Target: {lb_head})")
                print(f"  Right BC: {right_val:.4f} (Target: {rb_head})")
    
    print(f"\nBest Model:")
    print(f"Best loss: {min_loss:.4e} at iteration {best_iteration}")
    
    # Load best parameters
    if best_params:
        for name, param in piml.net.named_parameters():
            param.data.copy_(best_params[name])
    
    # Plot loss history
    plt.figure(figsize=(8, 6))
    plt.semilogy(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.grid(True)
    plt.savefig('loss_history.png')
    plt.close()
    
    # Generate solution for visualization
    x_plot = torch.linspace(xmin, xmax, 200).view(-1, 1).to(device)
    with torch.no_grad():
        h_plot = piml.predict(x_plot)
    
    # Plot the solution
    plt.figure(figsize=(8, 6))
    plt.plot(x_plot.cpu().numpy(), h_plot.cpu().numpy(), 'b-', linewidth=2, label='PINN Solution')
    plt.plot(x_left.cpu().numpy(), h_left.cpu().numpy(), 'ro', label='Left BC')
    plt.plot(x_right.cpu().numpy(), h_right.cpu().numpy(), 'go', label='Right BC')
    
    # Analytical solution for comparison
    h_analytical = lb_head + (rb_head - lb_head) * (x_plot - xmin) / (xmax - xmin)
    plt.plot(x_plot.cpu().numpy(), h_analytical.cpu().numpy(), 'r--', linewidth=2, label='Analytical Solution')
    
    plt.xlabel('$x$')
    plt.ylabel('$h(x)$')
    plt.title('1D Darcy Flow Solution')
    plt.legend()
    plt.grid(True)
    plt.savefig('darcy_solution.png')
    plt.show()
    
    # Calculate error metrics
    with torch.no_grad():
        h_pred = piml.predict(x_plot).cpu().numpy()
        h_true = h_analytical.cpu().numpy()
        
        mse_val = mean_squared_error(h_true, h_pred)
        r2_val = r2_score(h_true, h_pred)
        
        print(f"Mean Squared Error: {mse_val:.6e}")
        print(f"R² Score: {r2_val:.6f}")