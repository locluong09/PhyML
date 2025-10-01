import torch
import torch.nn as nn
import torch.optim as optim

from typing import List, Callable, Union, Optional

class PIMLModule(nn.Module):
    """Physics Informed Machine Learning Module
    This is a base class for creating physics-informed machine learning models.

    Args:
        net: machine learning model to train the physics-informed model
        pde_loss: function to compute the PDE loss
        optimizer: optimizer for the model
        lr: learning rate for the optimizer
        output_act: optional function to apply to the model output
        loss_fn: loss to be used ('mse' or 'sse')
        amp: whether to use automatic mixed precision
    """
    def __init__(
        self,
        net: nn.Module,
        pde_loss: Callable,
        optimizer: optim.Optimizer,
        lr: float = 0.001,
        output_act: Optional[Callable] = None,
        loss_fn: str = 'mse',
        amp: bool = False,
    ) -> None:
        super(PIMLModule, self).__init__()
        self.net = net
        self.pde_loss = pde_loss
        self.optimizer = optimizer(self.net.parameters(), lr = lr)
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        self.output_act = output_act
        if loss_fn == 'mse':
            self.loss_fn = mse
        elif loss_fn == "sse":
            self.loss_fn = sse
        else:
            raise ValueError(f'Unsupported loss function: {loss_fn}')
        
    def forward(self, spatial: Union[torch.Tensor, List[torch.Tensor]], time: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the machine learning network (net)

        Args:
            spatial: spatial coordinate tensor or list of spatial tensors
            time: temporal coordinate tensor
        
        Returns:
            output tensor from the model
        """
        # Handle spatial inputs as either a single tensor (e.g., 1D) or a list of tensors (2D or 3D)
        if isinstance(spatial, list):
            if time is not None:
                inputs = spatial + [time]
            else:
                inputs = spatial
        else:
            if time is not None:
                inputs = [spatial, time]
            else:
                inputs = [spatial]
        
        concat_inputs = torch.cat(inputs, dim = 1)
        outputs = self.net(concat_inputs)

        if self.output_act:
            if isinstance(spatial, list):
                if time is not None:
                    outputs = self.output_act(outputs, *spatial, time)
                else:
                    outputs = self.output_act(outputs, *spatial)
            else:
                if time is not None:
                    outputs = self.output_act(outputs, spatial, time)
                else:
                    outputs = self.output_act(output, spatial)
        
        return outputs
    
    def _get_device(self):
        """Get the device the model is on."""
        return next(self.parameters()).device

    def model_step(self, batch):
        """Perform a single model step on a batch of data
        
        Args:
            batch: A batch of data (a dictionary) mapping loss function names to input
            tensors for different conditions and data
        
        Returns:
            A tuple containing:
                - A tensor of total loss
                - A dictionary of predictions
        """
        # Initialized total loss and predictions
        total_loss = torch.tensor(0.0, device = self._get_device())
        all_predictions = {}

        # Process each component in a batch of data
        for name, data in batch.items():
            # unpacking data from batch items
            if name == "data_loss":
                if len(data) == 3:
                    spatial, time, target = data

                    if not isinstance(spatial, list):
                        spatial = [spatial]
                
                    for i in range(len(spatial)):
                        if not spatial[i].requires_grad:
                            spatial[i].requires_grad_(True)

                    if not time.requires_grad:
                        time.requires_grad_(True)
                
                    # get model predictions
                    predictions = self.forward(spatial, time)
            
                else:
                    spatial, target = data

                    if not isinstance(spatial, list):
                        spatial = [spatial]
                    
                    for i in range(len(spatial)):
                        if not spatial[i].requires_grad:
                            spatial[i].requires_grad_(True)
                    # get model predictions
                    predictions = self.forward(spatial)

                component_loss = self.loss_fn(predictions, target)
                all_predictions["data"] = predictions

            elif name == "pde_loss":
                # Handle PDE physics loss
                if len(data) > 1 and isinstance(data[-1], torch.Tensor) and data[-1].dim() > 0:  # Check if last item is a tensor (time)
                    spatial, time = data[0], data[1]
                    
                    # Ensure spatial is a list of tensors
                    if not isinstance(spatial, list):
                        spatial = [spatial]
                    
                    # Ensure all tensors require gradient for autograd
                    for i in range(len(spatial)):
                        if not spatial[i].requires_grad:
                            spatial[i].requires_grad_(True)
                    
                    if not time.requires_grad:
                        time.requires_grad_(True)
                    
                    # Use the dedicated PDE_loss function provided in __init__
                    component_loss = self.pde_loss(spatial, time, self)
                else:  # Only spatial (no time)
                    spatial = data[0]
                    
                    # Ensure spatial is a list of tensors
                    if not isinstance(spatial, list):
                        spatial = [spatial]
                    
                    # Ensure all tensors require gradient for autograd
                    for i in range(len(spatial)):
                        if not spatial[i].requires_grad:
                            spatial[i].requires_grad_(True)
                    
                    # Use the dedicated PDE_loss function provided in __init__
                    component_loss = self.pde_loss(spatial, None, self)
                
                all_predictions["pde"] = component_loss
                
            elif name == "boundary_loss":
                # Handle boundary condition loss
                if len(data) == 3:  # Spatial + time + boundary values
                    spatial, time, boundary_values = data
                    
                    # Ensure spatial is a list of tensors
                    if not isinstance(spatial, list):
                        spatial = [spatial]
                    
                    # Calculate boundary loss
                    predictions = self.forward(spatial, time)
                else:  # Only spatial + boundary values (no time)
                    spatial, boundary_values = data
                    
                    # Ensure spatial is a list of tensors
                    if not isinstance(spatial, list):
                        spatial = [spatial]
                    
                    # Calculate boundary loss
                    predictions = self.forward(spatial)
                
                component_loss = self.loss_fn(predictions, boundary_values)
                all_predictions["boundary"] = predictions
                
            elif name == "initial_loss":
                # Handle initial condition loss
                spatial, time, initial_values = data
                
                # Ensure spatial is a list of tensors
                if not isinstance(spatial, list):
                    spatial = [spatial]
                
                # Calculate initial condition loss
                predictions = self.forward(spatial, time)
                component_loss = self.loss_fn(predictions, initial_values)
                all_predictions["initial"] = predictions
                
            else:
                raise ValueError(f"Unknown loss function type: {name}")
            
            # Accumulate loss
            total_loss += component_loss
        
        return total_loss, all_predictions
    

    def train_step(self, batch):
        """Perform a single training step on a batch of data
        
        Args:
            batch: A batch of data for training
        
        Returns:
            The calculated loss
        """
        self.optimizer.zero_grad()
        if self.scaler:
            with torch.cuda.amp.autocast():
                loss, _ = self.model_step(batch)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss, _ = self.model_step(batch)
            loss.backward()
            self.optimizer.step()

        return loss.item()
    
    def val_step(self, batch):
        """Perform a single validation step on a batch of data
        
        Args:
            batch: A batch of data for validation
        
        Returns:
            The calculated loss and predictions
        """
        with torch.no_grad():
            loss, pred = self.model_step(batch)
        
        return loss.item(), pred
    
    def predict(self, spatial, time = None):
        """Make predictions using the trained model
        
        Args:
            spatial: spatial coordinate tensor or list of spatial tensors
            time: temporal coordinate tensor (optional)
        
        Returns:
            output tensor from the model
        """
        with torch.no_grad():
            predictions = self.forward(spatial, time)
        
        return predictions
    
def mse(y_true, y_pred):
    """Mean squared error loss function."""
    return torch.mean((y_true - y_pred)**2)


def sse(y_true, y_pred):
    """Sum of squared errors loss function."""
    return torch.sum((y_true - y_pred)**2)


