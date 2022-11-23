import math
import torch
import torch.nn as nn
import copy
from .layers import MultiHeadAttention, Embeddings, PositionwiseFeedForward, Generator, PositionalEncoding
from .encoder.layers import Encoder, EncoderLayer
from .decoder.layers import Decoder, DecoderLayer


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """
    def __init__(self, encoder, decoder, decoder_out, src_embed, tgt_embed,
                 generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_out = decoder_out  # final layer that converts d_model --> d_output
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        decoder_output = self.decode(self.encode(src, src_mask), src_mask, tgt,
                                     tgt_mask)
        output = self.decoder_out(decoder_output)
        output = torch.sigmoid(output)
        return output

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_time_series_model(d_input,
                           d_output,
                           N=4,
                           d_model=64,
                           d_ff=2048,
                           h=8,
                           dropout=0.2,
                           device="cpu"):
    """
    Make time series model with encoder and decoder

    :param d_input: Input dimensionality
    :param d_output: Output dimensionality
    :param N: Model depth
    :param d_model: Model dimensionality
    :param d_ff: Feedforward layer latent dimensionality
    :param h: number of heads
    :param dropout: dropout
    :param device: pytorch device
    :return: Transformer pytorch model
    """
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        decoder=Decoder(
            DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        decoder_out=nn.Linear(d_model, d_output),
        src_embed=nn.Sequential(nn.Linear(d_input, d_model), c(position)),
        tgt_embed=nn.Sequential(nn.Linear(d_output, d_model), c(position)),
        generator=Generator(d_model, d_output))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model.to(device)
    return model


class EncoderOnlyTransformer(nn.Module):
    """
    Encoder Only Transformer Model.

    """
    def __init__(self, decoder, decoder_out, src_embed):
        """
        :param decoder: decoder (or encoder as we only use one). Takes as input the embedded time series and performs one forward pass of the Transformer decoder
        :param decoder_out: decoder (or encoder) output layer. Layer that comes after N decoders that will be used as prediction layer
        :param src_embed: Embedding layer. Takes time series as input and outputs embedded time series
        """
        super(EncoderOnlyTransformer, self).__init__()
        self.decoder = decoder
        self.decoder_out = decoder_out  # final layer that converts d_model --> d_output
        self.src_embed = src_embed

    def decode(self, src, src_mask):
        return self.decoder(self.src_embed(src), src_mask)

    def forward(self, src, src_mask):
        decoder_output = self.decode(src, src_mask)
        output = self.decoder_out(decoder_output)
        return output


def make_encoder_only_time_series_model(d_input,
                                        d_output,
                                        N=12,
                                        d_model=64,
                                        d_ff=2048,
                                        h=8,
                                        dropout=0.2,
                                        device="cpu"):
    """
    Encoder only Transformer

    :param d_input: Input dimensionality
    :param d_output: Output dimensionality
    :param N: Model depth
    :param d_model: Model dimensionality
    :param d_ff: Feedforward layer latent dimensionality
    :param h: number of heads
    :param dropout: dropout
    :param device: pytorch device
    :return: Transformer pytorch model
    """

    # initialize layers
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    # construct model
    model = EncoderOnlyTransformer(
        decoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        decoder_out=nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_output)),  # dropout + dense output layer
        src_embed=nn.Sequential(
            nn.Linear(d_input, d_model),
            c(position)),  # positional embedding, by adding sinusoidal
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model.to(device)
    return model


class EncoderToDecoderTransformer(nn.Module):
    """
    Encoder output is directly used as input of the Decoder based on the work from [7]
    """
    def __init__(self, encoder, decoder, decoder_out, src_embed, tgt_embed,
                 generator):
        super(EncoderToDecoderTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_out = decoder_out  # final layer that converts d_model --> d_output
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, src_mask):
        """Take in and process masked src and target sequences.

        src: src
        src_mask: masking of source. Usually for this kind of architecture it should be all ones.
        """
        encoder_out = self.encoder(self.src_embed(src), src_mask)
        memory = encoder_out
        #encoder_out = self.tgt_embed(encoder_out)
        decoder_out = self.decoder(
            encoder_out, memory, src_mask,
            src_mask)  # No difference between src and tgt mask
        output = self.decoder_out(decoder_out)
        output = torch.sigmoid(output)
        return output


def make_encoder_to_decoder_time_series_model(d_input,
                                              d_output,
                                              N=12,
                                              d_model=64,
                                              d_ff=2048,
                                              h=8,
                                              dropout=0.2,
                                              device="cpu"):
    """
    Based on the architecture used by https://timeseriestransformer.readthedocs.io/en/latest/README.html.
    Basically the output of the encoder is used directly as input of the decoder.
    """

    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderToDecoderTransformer(
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        decoder=Decoder(
            DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        decoder_out=nn.Linear(d_model, d_output),
        src_embed=nn.Sequential(nn.Linear(d_input, d_model), c(position)),
        tgt_embed=nn.Sequential(nn.Linear(d_output, d_model), c(position)),
        generator=Generator(d_model, d_output))
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model.to(device)
    return model
