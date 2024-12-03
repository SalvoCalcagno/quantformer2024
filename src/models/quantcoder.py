import torch
import torch.nn as nn
from torch.autograd import Function
from src.models.quantformer import *

class PatchClassificationHead(nn.Module):
    def __init__(self, d_model, n_classes, head_dropout):
        super().__init__()
        self.dropout = nn.Dropout(head_dropout)
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, x):
        """
        x: [bs x nvars x d_model x num_patch]
        output: [bs x nvars x num_patch x n_classes]
        """
        # Tranpose
        #x = x.permute(0, 1, 3, 2) # x: [bs x nvars x num_patch x d_model]

        # Dropout
        x = self.dropout(x)
        
        # Linear Projection
        y = self.linear(x) # y: [bs x nvars x num_patch x n_classes]
        
        # You can compute predicted class probabilities by applying a softmax function to the output along the class (last) dimension
            
        return y

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
            num_classes = 512,
            freeze_encoder = False,
            cls = False,
            use_neuron_embedding = True,
            ablate_quantizer = False,
            stim_size=4,
            #**kwargs
        )
        
        # update the default args_dict with the user provided arguments
        for arg, default in args_defaults.items():
            setattr(self, arg, configs[arg] if arg in configs and configs[arg] is not None else default)
        
        self.stimulus_embedding = nn.Linear(self.stim_size, self.d_model)
        
        self.neuron_embedding = nn.Embedding(self.c_in, self.d_model)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        
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
            norm = self.norm, #"BatchNorm",
            cls=self.cls,
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
        
        # Disable gradients for Codebook
        for param in self.codebook.parameters():
            param.requires_grad = False
            
        if self.freeze_encoder:
            
            # Disable gradients for stimulus embedding
            for param in self.stimulus_embedding.parameters():
                param.requires_grad = False
                
            # Disable gradients for encoder
            for param in self.encoder.parameters():
                param.requires_grad = False

        if not self.ablate_quantizer:
            self.classification_head = PatchClassificationHead(self.d_model, self.num_classes, self.dropout)
        else:
            self.regression_head = PatchClassificationHead(self.d_model, self.patch_len, self.dropout)
        
    def encode(self, x, stim=None):
        if self.verbose:
            print(f"[PatchTST-VQVAE] x: {x.shape}]")
        if stim is not None:
            stimulus_embedding = self.stimulus_embedding(stim)
            if self.verbose:
                print(f"[PatchTST-VQVAE] stimulus_embedding: {stimulus_embedding.shape}]")
        else:
            stimulus_embedding = None
        z_e_x = self.encoder(x, stim = stimulus_embedding)
        if self.verbose:
            print(f"[PatchTST-VQVAE] x: {z_e_x.shape}]")
        indices = self.codebook(z_e_x)
        return indices

    def forward(self, x, stimulus, mask=None):
        """
        Parameters
        ----------
        x: tensor [bs x num_patch x nvars x patch_len]
        stimulus: tensor [bs]
        
        Returns
        -------
        x_tilde: tensor [bs x num_patch x nvars x patch_len]
        z_e_x: tensor [bs x nvars x d_model x num_patch]
        z_q_x: tensor [bs x nvars x d_model x num_patch]
        indices: tensor [bs x nvars x num_patch]
        """
        
        bs, num_patch, nvars, patch_len = x.shape
        if self.verbose:
            print(f"[QuantFormer] input: {x.shape}]")  
        
        # Stimulus embedding
        stimulus_embedding = self.stimulus_embedding(stimulus) # stimulus_embedding: [bs x d_model]
        # Unsqueeze Patch dimension
        stimulus_embedding = stimulus_embedding.unsqueeze(1)
        # Unsqueeze nvars dimension
        stimulus_embedding = stimulus_embedding.unsqueeze(2)
        # Expand 
        stimulus_embedding = stimulus_embedding.expand(bs, 1, nvars, self.d_model)
        if self.verbose:
            print(f"[QuantCoder] stimulus_embedding: {stimulus_embedding.shape}]")

        if self.use_neuron_embedding:
            # Neuron embedding
            neuron_embedding = torch.arange(nvars).unsqueeze(0).expand(bs, nvars).long().to(x.device) # neuron_embedding: [bs x nvars]
            neuron_embedding = self.neuron_embedding(neuron_embedding) # neuron_embedding: [bs x nvars x d_model]
            # Unsqueeze Patch dimension
            neuron_embedding = neuron_embedding.unsqueeze(1) # neuron_embedding: [bs x 1 x nvars x d_model]
            if self.verbose:
                print(f"[QuantCoder] neuron_embedding: {neuron_embedding.shape}]")
                
            # Concatenate Prompts
            prompts = torch.cat([stimulus_embedding, neuron_embedding], dim=1)
        else:
            prompts = stimulus_embedding
            
        if self.cls:
            # Add cls token
            cls_token = self.cls_token.expand(bs, 1, nvars, self.d_model)
            prompts = torch.cat([cls_token, prompts], dim=1)
        if self.verbose:
            print(f"[QuantCoder] prompts: {prompts.shape}]")
        
        # Encode
        z_e_x = self.encoder(x, prompts=prompts, mask=mask) # z_e_x: [bs x nvars x d_model x num_patch]
        if self.verbose:
            print(f"[QuantCoder] encoded input: {z_e_x.shape}]")
            
        if self.cls:
            # with cls flag z_e_c only contains prompt embeddings
            # Predict logits
            # get only the cls token (first token)
            cls_embedding = z_e_x[:, :, 0, :]
            logits = self.classification_head(cls_embedding)
            indices = None
        
        else:
            if self.ablate_quantizer:
                indices = None
                logits = self.regression_head(z_e_x) # logits: [bs x nvars x num_patch x patch_len]
                logits = logits.permute(0, 2, 1, 3) # logits: [bs x num_patch x nvars x patch_len]
            else:
                # Quantize
                _, indices = self.codebook(z_e_x) # indices: [bs x nvars x num_patch]
                indices = indices['q']
                if self.verbose:
                    print(f"[QuantCoder] indices: {indices.shape}]")
                    
                # Predict logits
                logits = self.classification_head(z_e_x) # logits: [bs x nvars x num_patch x n_classes]
            
        # You can consider [:, :, :num_src_patch] as indices
        # and [:, :, num_src_patch:, :] as logits 
        return indices, logits