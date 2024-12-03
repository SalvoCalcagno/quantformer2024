import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.autoformer_blocks.embed import DataEmbedding, DataEmbedding_wo_pos
from src.models.autoformer_blocks.autocorrelation import AutoCorrelation, AutoCorrelationLayer
from src.models.autoformer_blocks.encdec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    """

    def __init__(self, args):
        
        super(Model, self).__init__()
        
        args_defaults = dict(
            task_name = 'short_term_forecast',
            seq_len = 96,
            label_len = 48,
            pred_len = 24,
            output_attention = False,
            moving_avg = 3,
            enc_in = 7,
            d_model = 128,
            embed = 'fixed',
            freq = 'a',
            factor = 5,
            dropout = 0.6,
            n_heads = 8,
            d_ff = 512,
            e_layers = 3,
            d_layers = 2,
            activation = 'gelu',
            dec_in = 7,
            c_out = 7,
            stim_size = 4,
            cls=False,
        )
        
        for arg, default in args_defaults.items():
            setattr(self, arg, args[arg] if arg in args and args[arg] is not None else default)

        if self.cls:
            self.task_name = 'classification'
            self.num_class = args['num_classes']
            
        # Decomp
        kernel_size = self.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.d_model, self.embed, self.freq,
                                                  self.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, self.factor, attention_dropout=self.dropout,
                                        output_attention=self.output_attention),
                        self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    moving_avg=self.moving_avg,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding_wo_pos(self.dec_in, self.d_model, self.embed, self.freq,
                                                      self.dropout)
            
            self.stim_embedding = nn.Linear(self.stim_size, self.d_model)
            
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, self.factor, attention_dropout=self.dropout,
                                            output_attention=False),
                            self.d_model, self.n_heads),
                        AutoCorrelationLayer(
                            AutoCorrelation(False, self.factor, attention_dropout=self.dropout,
                                            output_attention=False),
                            self.d_model, self.n_heads),
                        self.d_model,
                        self.c_out,
                        self.d_ff,
                        moving_avg=self.moving_avg,
                        dropout=self.dropout,
                        activation=self.activation,
                    )
                    for l in range(self.d_layers)
                ],
                norm_layer=my_Layernorm(self.d_model),
                #projection=nn.Linear(self.d_model, self.c_out, bias=True),
                #projection=nn.Linear(self.d_model*2, self.c_out, bias=True)
            )
            
            self.projection = nn.Linear(self.d_model, self.c_out, bias=True)
            self.projection_with_stim = nn.Linear(self.d_model*2, self.c_out, bias=True)
            
        if self.task_name == 'imputation':
            self.projection = nn.Linear(
                self.d_model, self.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                self.d_model, self.c_out, bias=True)
        if self.task_name == 'classification':
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
            self.stim_embedding = nn.Linear(self.stim_size, self.d_model)
            self.act = F.gelu
            self.dropout = nn.Dropout(self.dropout)
            self.projection = nn.Linear(
                self.d_model, self.c_out)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, stim=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,
                             x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        
        if stim is not None:
            # late fusion stim
            stim = self.stim_embedding(stim)
            # expand
            stim = stim.unsqueeze(1).expand(-1, seasonal_part.shape[1], -1)
            # concat
            seasonal_part = torch.cat((seasonal_part, stim), dim=-1)
            # projection
            seasonal_part = self.projection_with_stim(seasonal_part)
        else:
            seasonal_part = self.projection(seasonal_part)
        
        # final
        dec_out = trend_part + seasonal_part
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc, stim=None):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        cls_token = self.cls_token.expand(enc_out.size(0), -1, -1)
        stim = self.stim_embedding(stim).unsqueeze(1)
        enc_out = torch.cat((cls_token, stim, enc_out), dim=1)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # select CLS
        enc_out = enc_out[:, 0, :]

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        #output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        #output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, stim=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, stim=stim)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc, stim=stim)
            return dec_out  # [B, N]
        return None
    
    def loss(self, output, target):
        if self.cls:
            return nn.BCEWithLogitsLoss()(output, target)
        else:
            return nn.MSELoss()(output, target)