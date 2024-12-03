from torchmetrics.regression import PearsonCorrCoef

class TimeSeriesPearsonCorrCoef():
        
    def __call__(self, x_tilde, x, patched=True):
        """
        Parameters
        ----------
        x_tilde: torch.Tensor
            The predicted values of shape [bs x num_patch x nvars x patch_len]
        x: torch.Tensor
            The target values of shape [bs x num_patch x nvars x patch_len]
        patched: bool 
            Whether to compute the correlation for each patch or for the entire sequence
            
        Returns
        -------
        correlation: torch.float
            The mean correlation among the patches or the entire sequence
        """
        
        # Get the shape of the input tensors
        bs, num_patch, nvars, patch_len = x_tilde.shape

        
        if patched:
            # Treat each patch as a sample
            x = x.contiguous().view(bs*num_patch*nvars, patch_len).t()
            x_tilde = x_tilde.contiguous().view(bs*num_patch*nvars, patch_len).t()
            
            # Remove padded rows 
            # x (window_size, num_rows)
            pad = (x == -100).all(0)
            x = x[:, ~pad]
            x_tilde = x_tilde[:, ~pad]
            num_outputs = x_tilde.shape[1]
            
            # Create correlation metric
            pearson = PearsonCorrCoef(num_outputs=num_outputs).to(x.device)
        
        else:
            # Reconstruct the original sequence
            x_tilde = x_tilde.permute(0, 2, 1, 3).reshape(bs, nvars, -1)
            x = x.permute(0, 2, 1, 3).reshape(bs, nvars, -1)
            
            # Treat timepoints as features and variables as samples
            x = x.contiguous().view(bs*nvars, -1).t()
            x_tilde = x_tilde.contiguous().view(bs*nvars, -1).t()
            
            # Remove padded rows 
            # x (window_size, num_rows)
            pad = (x == -100).all(0)
            x = x[:, ~pad]
            x_tilde = x_tilde[:, ~pad]
            num_outputs = x_tilde.shape[1]
            
            # Create correlation metric
            pearson = PearsonCorrCoef(num_outputs=num_outputs).to(x.device)
            
        # Compute the correlation
        correlation = pearson(x_tilde, x)
        
        # Return the mean correlation
        return correlation.abs().mean()
    
class CorrelationLoss():
        
    def __call__(self, x_tilde, x, patched=True):
        """
        Parameters
        ----------
        x_tilde: torch.Tensor
            The predicted values of shape [bs x num_patch x nvars x patch_len]
        x: torch.Tensor
            The target values of shape [bs x num_patch x nvars x patch_len]
        patched: bool 
            Whether to compute the correlation for each patch or for the entire sequence
            
        Returns
        -------
        correlation: torch.float
            The mean correlation among the patches or the entire sequence
        """
        
        # Get the shape of the input tensors
        bs, num_patch, nvars, patch_len = x_tilde.shape
        
        if patched:
            # Treat each patch as a sample
            x = x.contiguous().view(bs*num_patch*nvars, patch_len).t()
            x_tilde = x_tilde.contiguous().view(bs*num_patch*nvars, patch_len).t()
            
            # Create correlation metric
            pearson = PearsonCorrCoef(num_outputs=bs*num_patch*nvars).to(x.device)
        else:
            # Reconstruct the original sequence
            x_tilde = x_tilde.permute(0, 2, 1, 3).reshape(bs, nvars, -1)
            x = x.permute(0, 2, 1, 3).reshape(bs, nvars, -1)
            
            # Treat timepoints as features and variables as samples
            x = x.contiguous().view(bs*nvars, -1).t()
            x_tilde = x_tilde.contiguous().view(bs*nvars, -1).t()
            
            # Create correlation metric
            pearson = PearsonCorrCoef(num_outputs=bs*nvars).to(x.device)
            
        # Compute the correlation
        correlation = pearson(x_tilde, x)
        
        # Return the mean correlation
        return correlation.abs().mean()