import torch
import torch.nn as nn

# define model
class Model(nn.Module):

    def __init__(self, args):
        
        args_default = {
            'input_size': 1,
            'hidden_size': 32,
            'num_layers': 2,
            'dropout': 0.1,
            'batch_first': True,
            'output_size': 1,
            'verbose': False,
            'output_sequence_length': 64,
            'stim_size': 4,
            'cls': False,  
        }

        for arg, default in args_default.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
            
        super(Model, self).__init__()

        self.lstm = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            batch_first=self.batch_first,
            dropout=self.dropout,
            num_layers=self.num_layers
        )
        
        self.stim_embedding = nn.Linear(self.stim_size, self.hidden_size)
        
        if self.stim_size > self.hidden_size:
            self.linear = nn.Linear(self.hidden_size + self.stim_size, self.output_size)
        else: 
            self.linear = nn.Linear(self.hidden_size*2, self.output_size)

    def forward(self, src, tgt, stim):
        
        """
        Parameters
        ----------
        src: torch.Tensor
            Source sequence (batch, nvars, src_len)
        tgt: torch.Tensor
            Target sequence (batch, nvars, tgt_len)
            
        Returns
        -------
        output: torch.Tensor
            Output sequence (batch, nvars, tgt_len)
        """
        
        # Concatenate source and target sequences
        x = torch.cat([src, tgt[:, :, :-1]], dim=2)
        
        # Reshape to (batch, seq_len, input_size)
        x = x.transpose(1, 2)  

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        self.hidden = (h0, c0)

        x, h = self.lstm(x)

        if self.verbose:
            print(f"X shape: {x.shape}")
            print(f"H shape: {h[0].shape}")
            print(f"C shape: {h[1].shape}")
        
        # select the last N hidden states
        N = tgt.shape[2]
        x = x[:, -N:, :]
        
        # Extract stim features
        if self.stim_size < self.hidden_size:
            stim = self.stim_embedding(stim)

        # Concatenate Stim Features along the last dimension
        x = torch.cat([x, stim.unsqueeze(1).repeat(1, N, 1)], dim=-1)
        
        output = self.linear(x)
    
        # Reshape to (batch, nvars, seq_len)
        output = output.transpose(1, 2)

        return output
    
    def loss(self, output, target):
        return nn.MSELoss()(output, target)