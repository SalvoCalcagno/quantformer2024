import torch

def compute_psnr_4d(prediction, target):
    """
    Compute the PSNR between two tensors
    
    Parameters 
    ----------
    prediction: torch.Tensor
        It is the reconstructed tensor of shape (batch_size, nvars, num_patch, patch_len)
    target: torch.Tensor
        It is the original tensor of shape (batch_size, nvars, num_patch, patch_len)
        
    Returns
    -------
    psnr: torch.Tensor
        It is the PSNR between the two tensors of shape (batch_size, nvars)
    """

    bs, _, nvars, _ = target.shape
    
    prediction = prediction.permute(0, 2, 1, 3)
    target = target.permute(0, 2, 1, 3)
    
    # Get the MSE
    mse = ((target.reshape(bs, nvars, -1) - prediction.reshape(bs, nvars, -1)) ** 2).mean(dim=-1)
    
    # Get RMSE
    rmse = torch.sqrt(mse)
    
    # Get Maximum for each reconstruction
    max_f = torch.max(target.reshape(bs, nvars, -1), dim=2)[0]
    
    return 20*torch.log10((max_f/rmse).mean())

def compute_psnr_3d(prediction, target):
    """
    Compute the PSNR between two tensors
    
    Parameters 
    ----------
    target: torch.Tensor
        It is the original tensor of shape (batch_size, nvars, seq_len)
    prediction: torch.Tensor
        It is the reconstructed tensor of shape (batch_size, nvars, seq_len)
        
    Returns
    -------
    psnr: torch.Tensor
        It is the PSNR between the two tensors of shape (batch_size, nvars)
    """

    bs, nvars, _ = target.shape
    
    # Get the MSE
    mse = ((target - prediction) ** 2).mean(dim=-1)
    
    # Get RMSE
    rmse = torch.sqrt(mse)
    
    # Get Maximum for each reconstruction
    max_f = torch.max(target, dim=2)[0]
    
    return 20*torch.log10((max_f/rmse).mean())

def compute_psnr_2d(prediction, target):
    """
    Compute the PSNR between two tensors
    
    Parameters 
    ----------
    target: torch.Tensor
        It is the original tensor of shape (nvars, seq_len)
    prediction: torch.Tensor
        It is the reconstructed tensor of shape (nvars, seq_len)
        
    Returns
    -------
    psnr: torch.Tensor
        It is the PSNR between the two tensors of shape (nvars)
    """

    nvars, _ = target.shape
    
    # Get the MSE
    mse = ((target - prediction) ** 2).mean(dim=-1)
    
    # Get RMSE
    rmse = torch.sqrt(mse)
    
    # Get Maximum for each reconstruction
    max_f = torch.max(target, dim=-1)[0]
    
    return 20*torch.log10((max_f/rmse).mean())
    