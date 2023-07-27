import torch
import torch.nn as nn

class MultiNoiseLoss(nn.Module):
    def __init__(self, n_losses):
        """
        Initialise the module, and the scalar "noise" parameters (sigmas in arxiv.org/abs/1705.07115).
        If using CUDA, requires manually setting them on the device, even if the model is already set to device.
        """
        super(MultiNoiseLoss, self).__init__()
        
        if torch.cuda.is_available():
            self.noise_params = nn.Parameter(torch.rand(n_losses),requires_grad=True) 
        else:
            self.noise_params = nn.Parameter(torch.rand(n_losses),requires_grad=True)
    
    def forward(self, losses):
        """
        Computes the total loss as a function of a list of classification losses.
        TODO: Handle regressions losses, which require a factor of 2 (see arxiv.org/abs/1705.07115 page 4)
        """
        
        total_loss = 0
        for i, loss in enumerate(losses):
            total_loss += (1/torch.square(self.noise_params[i]))*loss + torch.log(self.noise_params[i])
        
        return total_loss