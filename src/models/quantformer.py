import torch
import torch.nn as nn
from torch.autograd import Function
from src.models.patchtst_self import *
from vqtorch.nn import VectorQuant


## PatchTST Backbone
class PatchTSTEncoder(nn.Module):
    
    def __init__(self, 
            c_in, 
            num_patch, 
            patch_len, 
            n_layers=3, 
            d_model=128, 
            n_heads=16, 
            shared_embedding=True,
            d_ff=256, 
            norm='BatchNorm', 
            attn_dropout=0., 
            dropout=0., 
            act="gelu", 
            store_attn=False,
            res_attention=True, 
            pre_norm=False,
            pe='zeros', 
            learn_pe=True, 
            verbose=False, 
            learnable_mask=False,
            cls = False,
            **kwargs
        ):
        
        """
        Parameters:
        -----------
        c_in: int
            number of variables or channels in the input (in our case it is the number of neurons)
        num_patch: int
            number of patches in the input. This count excludes the [STIM] token
        patch_len: int
            length of each patch (in our case it is the number of time points)
        n_layers: int
            number of layers in the encoder
        d_model: int
            dimension of the embedding space (token)
        n_heads: int
            number of heads in the multi-head attention
        shared_embedding: bool
            whether to share the embedding across the variables. If True, each variable is projected through a different linear layer
        d_ff: int
            dimension of the feed forward layer
        norm: str
            type of normalization to use. Can be "BatchNorm" or "LayerNorm"
        ----- auto-generated. don't rely on description -----
        attn_dropout: float
            dropout rate for the attention layer
        dropout: float
            dropout rate for the residual layers
        act: str
            activation function to use. Can be "relu" or "gelu"
        store_attn: bool
            whether to store the attention weights
        res_attention: bool
            whether to use residual attention
        pre_norm: bool
            whether to use pre-norm or post-norm
        pe: str
            type of positional encoding to use. Can be "zeros", "learnable", "sinusoidal"
        learn_pe: bool
            whether to learn the positional encoding
        verbose: bool
            whether to print the shape of the tensors at each layer
        learnable_mask: bool
            whether to learn a mask token to replace the masked values
        **kwargs: dict
            additional arguments
        """

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch # num patch must exclude the [STIM] token
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding      
        self.learnable_mask = learnable_mask
        self.cls = cls

        # Input encoding: projection of feature vectors onto a d-model vector space
        if not shared_embedding: 
            self.W_P = nn.ModuleList()
            for _ in range(self.n_vars): self.W_P.append(nn.Linear(patch_len, d_model))
        else:
            self.W_P = nn.Linear(patch_len, d_model)      

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch + 1, d_model)

        # [CLS] token
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # [MASK] token
        self.mask_token = nn.Parameter(torch.zeros(1, d_model))

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(
            d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
            pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
            store_attn=store_attn
        )
        
    def forward(self, x, prompts=None, mask=None) -> Tensor:          
        """
        Parameters
        ----------
        x: tensor [bs x num_patch x nvars x patch_len]
            The input time series.
        prompts: tensor [bs x nvars x num_prompts x d_model]
            The stimulus embedding [STIM]
        mask : tensor [bs x num_patch x nvars]
            The mask tensor. 1s indicate values that should be masked with the [MASK] token.
                
        Returns
        -------
        z: tensor [bs x nvars x num_patch x d_model]
        """
        
        # Get dimensions
        bs, num_patch, n_vars, patch_len = x.shape
        num_prompts = prompts.shape[1] if prompts is not None else 0
        
        # Input Embedding
        if not self.shared_embedding:
            x_out = []
            for i in range(n_vars): 
                z = self.W_P[i](x[:,:,i,:])
                x_out.append(z)
            x = torch.stack(x_out, dim=2)
        else:
            x = self.W_P(x) # x: [bs x num_patch x nvars x d_model]
        x = x.transpose(1,2) # x: [bs x nvars x num_patch x d_model]

        # Reshape - consider each variable as a separate sequence
        u = torch.reshape(x, (bs*n_vars, num_patch, self.d_model)) # u: [bs * nvars x num_patch x d_model]

        # Masking
        if mask is not None and self.learnable_mask:
            # replace masked values by the learned mask token
            # mask: [bs x num_patch x nvars]
            mask = mask.transpose(1,2) # mask: [bs x nvars x num_patch]
            mask = torch.reshape(mask, (bs*n_vars, num_patch)) # mask: [bs * nvars x num_patch] 
            mask_coords = torch.where(mask)
            u[mask_coords] = self.mask_token

        # Add CLS token
        #cls_tokens = self.cls_token.expand(bs*n_vars, -1, -1) # cls_tokens: [bs * nvars x 1 x d_model]
        #u = torch.cat((cls_tokens, u), dim=1) # u: [bs * nvars x (num_patch+1) x d_model]
        #num_patch = num_patch + 1
        
        # Add Positional Encoding
        u = self.dropout(u + self.W_pos[:num_patch, :]) # u: [bs * nvars x num_patch x d_model]
        
        # Add Prompts
        if prompts is not None:
            # Transpose prompts
            prompts = prompts.transpose(1,2) # prompts: [bs x nvars x num_prompts x d_model] 
            # Reshape prompts
            prompts = torch.reshape(prompts, (bs*n_vars, -1, self.d_model)) # prompts: [bs * nvars x num_prompts x d_model]
            # Concatenate prompts to the input
            u = torch.cat((prompts, u), dim=1) # u: [bs * nvars x (num_patch+num_prompts) x d_model]

        # Encoder
        z = self.encoder(u) # z: [bs * nvars x num_patch x d_model]
        z = torch.reshape(z, (-1, n_vars, num_patch+num_prompts, self.d_model)) # z: [bs x nvars x num_patch+num_prompts x d_model]
        #z = z.permute(0,1,3,2) # z: [bs x nvars x d_model x num_patch+num_prompts]
        
        # Remove prompts
        if prompts is not None:
            if self.cls:
                prompts = z[:, :, :num_prompts, :]
            z = z[:, :, num_prompts:, :]
        
        return prompts if self.cls else z

class PatchTSTDecoder(nn.Module):
    
    def __init__(self, 
        c_in, 
        num_patch, 
        patch_len, 
        n_layers=3, 
        d_model=128, 
        n_heads=16, 
        shared_embedding=True,
        d_ff=256, 
        norm='BatchNorm', 
        attn_dropout=0., 
        dropout=0., 
        act="gelu", 
        store_attn=False,
        res_attention=True, 
        pre_norm=False,
        pe='zeros', 
        learn_pe=True, 
        verbose=False, 
        learnable_mask=False, 
        **kwargs):
        
        """
        Parameters:
        -----------
        c_in: int
            number of variables or channels in the input (in our case it is the number of neurons)
        num_patch: int
            number of patches in the input. This count excludes the [STIM] token
        patch_len: int
            length of each patch (in our case it is the number of time points)
        n_layers: int
            number of layers in the encoder
        d_model: int
            dimension of the embedding space (token)
        n_heads: int
            number of heads in the multi-head attention
        shared_embedding: bool
            whether to share the embedding across the variables. If True, each variable is projected through a different linear layer
        d_ff: int
            dimension of the feed forward layer
        norm: str
            type of normalization to use. Can be "BatchNorm" or "LayerNorm"
        ----- auto-generated. don't rely on description -----
        attn_dropout: float
            dropout rate for the attention layer
        dropout: float
            dropout rate for the residual layers
        act: str
            activation function to use. Can be "relu" or "gelu"
        store_attn: bool
            whether to store the attention weights
        res_attention: bool
            whether to use residual attention
        pre_norm: bool
            whether to use pre-norm or post-norm
        pe: str
            type of positional encoding to use. Can be "zeros", "learnable", "sinusoidal"
        learn_pe: bool
            whether to learn the positional encoding
        verbose: bool
            whether to print the shape of the tensors at each layer
        learnable_mask: bool
            whether to learn a mask token to replace the masked values
        **kwargs: dict
            additional arguments
        """

        super().__init__()
        self.n_vars = c_in
        self.num_patch = num_patch # num patch must exclue the [STIM] token
        self.patch_len = patch_len
        self.d_model = d_model
        self.shared_embedding = shared_embedding      
        self.learnable_mask = learnable_mask  

        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, num_patch + 1, d_model)
        
        # [MASK] token
        self.mask_token = nn.Parameter(torch.zeros(1, d_model))

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Decoder
        self.decoder = TSTEncoder(
            d_model, n_heads, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
            pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, 
            store_attn=store_attn
        )
        
        # Output decoding: projection of d-model vector space onto the patch-len space
        self.W_P = nn.Linear(d_model, patch_len)      

    def forward(self, z, stim=None, mask=None) -> Tensor:          
        """
        Parameters
        ----------
        z: tensor [bs x nvars x num_patch x d_model]
            Input tensor (quantized time series)
        stim: tensor [bs x d_model]
            The stimulus embedding [STIM]
        mask : tensor [bs x num_patch x nvars]
            The mask tensor. 1s indicate values that should be masked with the [MASK] token.
        
        """
        
        # Get dimensions
        bs, nvars, num_patch, d_model = z.shape
        
        # Reshape - consider each variable as a separate sequence
        #z = z.permute(0,1,3,2) # x: [bs x nvars x num_patch x d_model]
        z = torch.reshape(z, (bs*nvars, num_patch, d_model) ) # x: [bs * nvars x num_patch x d_model]
        
        # Masking
        if mask is not None and self.learnable_mask:
            # replace masked values by the learned mask token
            # mask: [bs x num_patch x nvars]
            mask = mask.transpose(1,2) # mask: [bs x nvars x num_patch]
            mask = torch.reshape(mask, (bs*nvars, num_patch)) # mask: [bs * nvars x num_patch] 
            mask_coords = torch.where(mask)
            z[mask_coords] = self.mask_token
        
        # Add [STIM] token
        if stim is not None:
            # stim: [bs x d_model]
            # unsqueeze patch dimension
            stim = stim.unsqueeze(1) 
            # unsqueeze nvars dimension and expand
            stim = stim.unsqueeze(1).expand(-1, nvars, -1, -1) # stim: [bs x nvars x 1 x d_model]
            # reshape
            stim = torch.reshape(stim, (bs*nvars, 1, self.d_model)) # stim: [bs * nvars x 1 x d_model]
            # concat
            z = torch.cat((stim, z), dim=1) # u: [bs * nvars x (num_patch+1) x d_model]
            num_patch = num_patch + 1
        
        # Add Positional Encoding
        z = self.dropout(z + self.W_pos[:num_patch, :])                                         

        # Decoder
        x_tilde = self.decoder(z)
        
        # Project to the common space
        x_tilde = self.W_P(x_tilde) # x_tilde: [bs * nvars x num_patch x patch_len]
        
        # Reshape
        x_tilde = torch.reshape(x_tilde, (bs, nvars, num_patch, self.patch_len)) # x_tilde: [bs x nvars x num_patch x patch_len]
        #x_tilde = x_tilde.permute(0,1,3,2) # x_tilde: [bs x nvars x patch_len x num_patch]
        
        # Remove [STIM] token
        if stim is not None:
            x_tilde = x_tilde[:, :, 1:, :]

        return x_tilde

class Model(nn.Module):
    
    def __init__(self, configs):
        super().__init__()
        
        args_defaults=dict(
            c_in = 7,
            target_dim = 24,
            patch_len = 16,
            stride = 16,
            num_patch = 42,
            n_layers=3, 
            d_model=128, 
            n_heads=16, 
            shared_embedding=True, 
            d_ff=256, 
            norm='BatchNorm', 
            attn_dropout=0., 
            dropout=0., 
            act="gelu", 
            res_attention=True, 
            pre_norm=False, 
            store_attn=False,
            pe='zeros', 
            learn_pe=True, 
            head_dropout = 0, 
            head_type = "prediction", 
            individual = False, 
            y_range=None, 
            verbose=False,
            learnable_mask=False,
            K = 512, #number of codes,
            max_stimulus_index = 10,
            ablate_quantizer = False,
            #**kwargs
        )
        
        # update the default args_dict with the user provided arguments
        for arg, default in args_defaults.items():
            setattr(self, arg, configs[arg] if arg in configs and configs[arg] is not None else default)
        
        self.stimulus_embedding = nn.Embedding(self.max_stimulus_index, self.d_model)
        
        self.encoder = PatchTSTEncoder(
            c_in=self.c_in, 
            num_patch=self.num_patch, 
            patch_len=self.patch_len, 
            n_layers=self.n_layers, #2
            d_model=self.d_model, 
            n_heads=self.n_heads, #4
            shared_embedding=self.shared_embedding, #True
            d_ff=self.d_ff, #128
            attn_dropout=self.attn_dropout, #0
            dropout=self.dropout, #0
            act=self.act, #"gelu"
            res_attention=self.res_attention, #True
            pre_norm=self.pre_norm, #False
            store_attn=self.store_attn, #False
            pe=self.pe, #"zeros"
            learn_pe=self.learn_pe, #True
            verbose=self.verbose, 
            learnable_mask=self.learnable_mask, #False
            norm = self.norm #"BatchNorm"
        )

        self.codebook = VectorQuant(
            feature_size=self.d_model, 
            num_codes=self.K, 
            beta=0.25, 
            kmeans_init=False, 
            norm=None, 
            cb_norm=None, 
            affine_lr=10.0, 
            sync_nu=0.2, 
            replace_freq=20, 
            dim=-1
        )

        self.decoder = PatchTSTDecoder(
            c_in=self.c_in, 
            num_patch=self.num_patch, 
            patch_len=self.patch_len, 
            n_layers=self.n_layers, #2
            d_model=self.d_model, 
            n_heads=self.n_heads, #4
            shared_embedding=self.shared_embedding, #True
            d_ff=self.d_ff, #128
            attn_dropout=self.attn_dropout, #0
            dropout=self.dropout, #0
            act=self.act, #"gelu"
            res_attention=self.res_attention, #True
            pre_norm=self.pre_norm, #False
            store_attn=self.store_attn, #False
            pe=self.pe, #"zeros"
            learn_pe=self.learn_pe, #True
            verbose=self.verbose, 
            learnable_mask=self.learnable_mask, #False
            norm = self.norm #"BatchNorm"
        )

    def encode(self, x, stim=None, return_embeddings=False):
        if self.verbose:
            print(f"[PatchTST-VQVAE] x: {x.shape}]")
        if stim is not None:
            stimulus_embedding = self.stimulus_embedding(stim)
            if self.verbose:
                print(f"[PatchTST-VQVAE] stimulus_embedding: {stimulus_embedding.shape}]")
        else:
            stimulus_embedding = None
        
        z_e_x = self.encoder(x, prompts=stimulus_embedding)
        
        if self.verbose:
            print(f"[PatchTST-VQVAE] x: {z_e_x.shape}]")
        
        z_q, vq_dict = self.codebook(z_e_x)
        
        if return_embeddings:
            return z_q, vq_dict['q']
        else:
            return vq_dict['q']
        

    def decode(self, latents, stim = None):
        if stim is not None:
            stimulus_embedding = self.stimulus_embedding(stim)
            if self.verbose:
                print(f"[PatchTST-VQVAE] stimulus_embedding: {stimulus_embedding.shape}]")
        else:
            stimulus_embedding = None
        latents = latents.squeeze()
        # TODO: try get codebook
        z_q_x = self.codebook.codebook(latents)  # (B, D, H, W)
        # TODO: Verify Affine transform
        if self.verbose:
            print(f"[PatchTST-VQVAE] z_q_x: {z_q_x.shape}]")
        x_tilde = self.decoder(z_q_x, stim = stimulus_embedding)
        x_tilde = x_tilde.permute(0, 2, 1, 3)
        if self.verbose:
            print(f"[PatchTST-VQVAE] x_tilde: {x_tilde.shape}]")
        return x_tilde

    def forward(self, x, stimulus_id=None, mask=None):
        """
        Parameters
        ----------
        x: tensor [bs x num_patch x nvars x patch_len]
        stimulus_id: tensor [bs]
        
        Returns
        -------
        x_tilde: tensor [bs x num_patch x nvars x patch_len]
        z_e_x: tensor [bs x nvars x d_model x num_patch]
        z_q_x: tensor [bs x nvars x d_model x num_patch]
        indices: tensor [bs * nvars * num_patch]
        """
        
        if self.verbose:
            print(f"[QuantFormer] input: {x.shape}]")  
        
        # Stimulus embedding
        if stimulus_id is not None:
            stimulus_embedding = self.stimulus_embedding(stimulus_id) # stimulus_embedding: [bs x d_model]
            if self.verbose:
                print(f"[QuantFormer] stimulus_embedding: {stimulus_embedding.shape}]")
        else:
            stimulus_embedding = None
            
        # Encode
        z_e_x = self.encoder(x, prompts=stimulus_embedding, mask=mask) # z_e_x: [bs x nvars x num_patch x d_model]
        if self.verbose:
            print(f"[QuantFormer] encoded input: {z_e_x.shape}]")
        
        if self.ablate_quantizer:
            z_q_x = z_e_x
            bs, nvars, num_patch, _ = z_q_x.shape
            vq_dict = {
                'q': torch.zeros(bs, nvars, num_patch),
            }
        else:
            # Quantize
            z_q_x, vq_dict = self.codebook(z_e_x) # z_q_x: [bs x nvars x num_patch x d_model]
            if self.verbose:
                print(f"[QuantFormer] quantized input: {z_q_x.shape}]")
        
        # Decode
        x_tilde = self.decoder(z_q_x, stim=stimulus_embedding, mask=None) # x_tilde: [bs x nvars x num_patch x patch_len]
        
        # Restore original shape
        x_tilde = x_tilde.permute(0, 2, 1, 3) # x_tilde: [bs x num_patch x nvars x patch_len]
        if self.verbose:
            print(f"[QuantFormer] decoded input: {x_tilde.shape}]")
        
        return x_tilde, z_e_x, z_q_x, vq_dict