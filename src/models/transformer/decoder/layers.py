import torch.nn as nn
from ..utils import clones
from ..layers import LayerNorm, SublayerConnection


class Decoder(nn.Module):
    """Generic N layer decoder with masking. This means that the complete Decoder generally
    consists of N times the same layer. """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """One Decoder Layer. Made off of self attention, src attention and feed forward"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        :param size: used for layer normalization == number of features
        :param self_attn:
        :param src_attn:
        :param feed_forward:
        :param dropout:
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
