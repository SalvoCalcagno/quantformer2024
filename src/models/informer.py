import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.informer_blocks.masking import TriangularCausalMask, ProbMask
from src.models.informer_blocks.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from src.models.informer_blocks.decoder import Decoder, DecoderLayer
from src.models.informer_blocks.attn import FullAttention, ProbAttention, AttentionLayer
from src.models.informer_blocks.embed import DataEmbedding

class Model(nn.Module):
    def __init__(self, args):
        
        super(Model, self).__init__()

        args_defaults=dict(
            enc_in = 7, 
            dec_in = 7, 
            c_out = 7, 
            seq_len = 96, 
            label_len = 48, 
            pred_len = 24, 
            factor = 5, 
            d_model=512, 
            n_heads=8, 
            e_layers=3, 
            d_layers=2, 
            d_ff=512, 
            dropout=0.0, 
            attn='prob', 
            embed='fixed', 
            freq='h', 
            activation='gelu', 
            output_attention = False, 
            distil=True, 
            mix=True,
            device='cuda:0',
            stim_size=4,
            cls = False,
        )
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
        
        self.device = torch.device(self.device)
        #self.pred_len = self.out_len
        #self.attn = attn
        #self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq, self.dropout)
        self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq, self.dropout)
        # Attention
        Attn = ProbAttention if self.attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, self.factor, attention_dropout=self.dropout, output_attention=self.output_attention), 
                                self.d_model, self.n_heads, mix=False),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            [
                ConvLayer(
                    self.d_model
                ) for l in range(self.e_layers-1)
            ] if self.distil else None,
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, self.factor, attention_dropout=self.dropout, output_attention=False), 
                                self.d_model, self.n_heads, mix=self.mix),
                    AttentionLayer(FullAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False), 
                                self.d_model, self.n_heads, mix=False),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        
        if self.cls:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
            
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)
        self.stim_embedding = nn.Linear(self.stim_size, self.d_model)
        self.projection_with_stim = nn.Linear(self.d_model*2, self.c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
        enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, stim=None):
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        if self.cls:
            # Add CLS token
            cls_token = self.cls_token.expand(dec_out.size(0), -1, -1)
            dec_out = torch.cat((cls_token, dec_out), dim=1)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        
        if self.cls:
            # Use only the CLS token
            dec_out = dec_out[:,0,:].unsqueeze(1)
            
        if stim is not None:
            stim = self.stim_embedding(stim)
            # expand 
            stim = stim.unsqueeze(1).expand(-1, dec_out.size(1), -1)
            dec_out = torch.cat((dec_out, stim), dim=-1)
            dec_out = self.projection_with_stim(dec_out)

        else:
            dec_out = self.projection(dec_out)
        
        if self.cls:
            return dec_out.squeeze()
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]
        
    def loss(self, output, target):
        if self.cls:
            return nn.BCEWithLogitsLoss()(output, target)
        else:
            return nn.MSELoss()(output, target)

class InformerStack(nn.Module):
    def __init__(self, args):
        
        super(InformerStack, self).__init__()
        
        args_defaults=dict(
            enc_in = 7, 
            dec_in = 7, 
            c_out= 7, 
            seq_len = 96, 
            label_len= 48, 
            pred_len = 24, 
            factor=5, 
            d_model=512, 
            n_heads=8, 
            e_layers=[3,2,1], 
            d_layers=2, 
            d_ff=512, 
            dropout=0.0, 
            attn='prob', 
            embed='fixed', 
            freq='h', 
            activation='gelu',
            output_attention = False, 
            distil=True, 
            mix=True,
            device='cuda:0'
        )
        for arg,default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)
        
        self.device = torch.device(self.device)
        
        #self.pred_len = out_len
        #self.attn = attn
        #self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(self.enc_in, self.d_model, self.embed, self.freq, self.dropout)
        self.dec_embedding = DataEmbedding(self.dec_in, self.d_model, self.embed, self.freq, self.dropout)
        # Attention
        Attn = ProbAttention if self.attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(self.e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, self.factor, attention_dropout=self.dropout, output_attention=self.output_attention), 
                                    self.d_model, self.n_heads, mix=False),
                        self.d_model,
                        self.d_ff,
                        dropout=self.dropout,
                        activation=self.activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        self.d_model
                    ) for l in range(el-1)
                ] if self.distil else None,
                norm_layer=torch.nn.LayerNorm(self.d_model)
            ) for el in self.e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, self.factor, attention_dropout=self.dropout, output_attention=False), 
                                self.d_model, self.n_heads, mix=self.mix),
                    AttentionLayer(FullAttention(False, self.factor, attention_dropout=self.dropout, output_attention=False), 
                                self.d_model, self.n_heads, mix=False),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(self.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]
