import torch
import torch.nn as nn
from src.metrics.FocalLoss import FocalLoss

# define model
class Model(nn.Module):

    def __init__(self, args):
        
        args_default = {
            'num_layers': 2,
            'dropout': 0.1,
            'hidden_size': 32,
            'stim_size': 4,
            'verbose': False,
            "loss_fn": "cross_entropy",
            "device": "cpu",
        }

        for arg, default in args_default.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
            
        super(Model, self).__init__()
        
        # Set loss
        self.set_loss(self.loss_fn, device=self.device)

        # Temporal Encoder
        self.lstm = nn.LSTM(
            input_size=1, 
            hidden_size=self.hidden_size,
            batch_first=True,
            dropout=self.dropout,
            num_layers=self.num_layers
        )
        
        # Stim Embedding
        self.stim_embedding = nn.Linear(self.stim_size, self.hidden_size)
        
        # Spatial Encoder
        self.spatial_encoder = nn.Sequential(
            
            nn.Conv1d(self.hidden_size*2, self.hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size),
            
            nn.Conv1d(self.hidden_size, self.hidden_size//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size//2),
            
            nn.Conv1d(self.hidden_size//2, self.hidden_size//4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size//4)
        )
        
        # Final Classifier
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size//4, self.hidden_size//8),
            nn.ReLU(),
            nn.Linear(self.hidden_size//8, 2),
        )

    def forward(self, src, stim):
        
        """
        Parameters
        ----------
        src: torch.Tensor
            Source sequence (batch, nvars, src_len)
        stim: torch.Tensor
            Stimulus features (batch, stim_size)
            
        Returns
        -------
        output: torch.Tensor
            Output sequence (batch, nvars)
        """
        
        # Get src shape
        batch_size, nvars, src_len = src.shape
        if self.verbose:
            print(f"[LSTM Classification] batch_size: {batch_size}, nvars: {nvars}, src_len: {src_len}")
            print(f"[LSTM Classification] src shape: {src.shape}")
        
        # Reshape to have single temporal feature
        src = src.reshape(batch_size*nvars, src_len, 1)
        
        if self.verbose:
            print(f"[LSTM Classification] src shape: {src.shape}")

        # Encode temporal features
        x, (h, c) = self.lstm(src)

        if self.verbose:
            print(f"[LSTM Classification] x shape: {x.shape}")
            print(f"[LSTM Classification] h shape: {h.shape}")
            print(f"[LSTM Classification] c shape: {c.shape}")

        # Select the last hidden state and restore channels
        x = x[:, -1, :].reshape(batch_size, nvars, -1)
        
        if self.verbose:
            print(f"[LSTM Classification] x shape: {x.shape}")
        
        # Extract stim features
        stim = self.stim_embedding(stim)
        
        if self.verbose:
            print(f"[LSTM Classification] stim shape: {stim.shape}")
        
        # Concatenate Stim Features along the last dimension
        stim = stim.unsqueeze(1).repeat(1, nvars, 1)
        x = torch.cat([x, stim], dim=-1)
        
        if self.verbose:
            print(f"[LSTM Classification] x shape: {x.shape}")
            
        # Transpose to ectract features along channel dimension
        x = x.transpose(1, 2)
        
        if self.verbose:
            print(f"[LSTM Classification] x shape: {x.shape}")
        
        # Encode spatial features
        x = self.spatial_encoder(x)
        
        # Transpose
        x = x.transpose(1, 2)
        
        if self.verbose:
            print(f"[LSTM Classification] x shape: {x.shape}")
        
        # Classify Activation
        output = self.linear(x)
        
        if self.verbose:
            print(f"[LSTM Classification] output shape: {output.shape}")

        return output
    
    def set_loss(self, loss_fn='cross_entropy', device='cpu'):
        
        if loss_fn == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()
        
        elif loss_fn == 'focal':
            self.loss_fn = FocalLoss(
                alpha=torch.tensor([0.05, 0.95]).to(device),
                gamma=2.0,
                reduction='mean',
            )
            
        else:
            raise ValueError(f"Loss function {loss_fn} not supported")
    
    def loss(self, output, target):
        """
        Parameters
        ----------
        output: torch.Tensor
            Output sequence (batch, nvars, 2)
        target: torch.Tensor
            Target sequence (batch, nvars)
        """
        
        output = output.transpose(1, 2)
        return self.loss_fn(output, target)