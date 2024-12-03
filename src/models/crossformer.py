import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from src.models.crossformer_blocks.cross_encoder import Encoder
from src.models.crossformer_blocks.cross_decoder import Decoder
from src.models.crossformer_blocks.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from src.models.crossformer_blocks.cross_embed import DSW_embedding

from math import ceil

class Model(nn.Module):
    
    def __init__(self, args):

        super(Model, self).__init__()

        args_defaults=dict(
            data_dim=186, # dimensionality of data, i.e. number of channels/features or number of time series 
            in_len=800, # length of input time series (number of time steps - raw) 
            out_len = 200, # length of output time series (number of lookhaed time steps - raw) 
            seg_len = 24, # length of segment (number of time steps to aggregate - raw) 
            win_size = 4,
            factor=10, 
            d_model=512, 
            d_ff = 1024, 
            n_heads=8, 
            e_layers=3, 
            dropout=0.0, 
            baseline = False, 
            device='cuda:0',
            verbose=False,
            stim_size = 4,
            cls = False,
            num_classes = 2,
        )
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
        
        self.device = torch.device(self.device)

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * self.in_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * self.out_len / self.seg_len) * self.seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(self.seg_len, self.d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, (self.pad_in_len // self.seg_len), self.d_model))
        self.pre_norm = nn.LayerNorm(self.d_model)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))

        # Encoder
        self.encoder = Encoder(
            self.e_layers, 
            self.win_size, 
            self.d_model, self.n_heads, self.d_ff, 
            block_depth = 1,
            dropout = self.dropout,
            in_seg_num = (self.pad_in_len // self.seg_len),# + 1, 
            factor = self.factor
        )
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, (self.pad_out_len // self.seg_len), self.d_model))
        self.stim_embedding = nn.Linear(self.stim_size, self.d_model)
        
        self.decoder = Decoder(
            self.seg_len, 
            self.e_layers + 1, 
            self.d_model, self.n_heads, self.d_ff, 
            self.dropout,
            out_seg_num = (self.pad_out_len // self.seg_len), 
            factor = self.factor
        )
        
        if self.cls:
            # Classifier
            self.cls_head = nn.Linear(self.out_len, self.num_classes)
        
    def forward(self, x_seq, stim=None):
        """
        Parameters
        ----------
        x_seq: torch.Tensor
            Source sequence (batch, nvars, src_len)
        stim: torch.Tensor
            Stimulus (batch, stim_size)
        """
        
        # Reshape to (batch, seq_len, input_size)
        x_seq = x_seq.transpose(1, 2)
        
        
        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0    
        batch_size = x_seq.shape[0]
        
        if self.verbose:
            print("Input shape:", x_seq.shape)
            
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)
        
        if self.verbose:
            print("After padding:", x_seq.shape)
            
        x_seq = self.enc_value_embedding(x_seq)
        
        if self.verbose:
            print("After embedding:", x_seq.shape)
            
        x_seq += self.enc_pos_embedding
        
        if self.verbose:
            print("After adding positional embedding:", x_seq.shape)
            
        x_seq = self.pre_norm(x_seq)
        
        if self.verbose:
            print("After layer norm:", x_seq.shape)
            
        enc_out = self.encoder(x_seq)

        if self.verbose:
            print("After encoder:", [x.shape for x in enc_out])

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
      
        if stim is not None:
            stim = self.stim_embedding(stim)
            _, ts_d, l, _ = dec_in.shape
            stim = repeat(stim, 'b d -> b ts_d l d', ts_d = ts_d, l = l)
            dec_in = dec_in + stim
        
        if self.verbose:
            print("After repeat:", dec_in.shape)
            
        predict_y = self.decoder(dec_in, enc_out)
        
        if self.verbose:
            print("After decoder:", predict_y.shape)
            
        if self.cls:
            output = self.cls_head(predict_y[:, :self.out_len, :].permute(0, 2, 1))
            return output
            
        output = base + predict_y[:, :self.out_len, :]
        
        # Reshape to (batch, nvars, seq_len)
        output = output.transpose(1, 2)
            
        return output
    
    def loss(self, output, target):
        if self.cls:
            return nn.CrossEntropyLoss()(output, target)
        else:
            return nn.MSELoss()(output, target)