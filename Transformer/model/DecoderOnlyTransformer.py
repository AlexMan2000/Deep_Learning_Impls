from common_import import *
from utils import *

class DecoderOnlyTransformer(nn.Module):

    def __init__(self,
                 tgt_vocab_size,
                 d_model,
                 d_ff,
                 num_head,
                 num_layers,
                 dropout=0.1
                 ):
        super().__init__()
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.d_model = d_model

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_head,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=d_ff
        )
        self.decoder_layers = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )

    def forward(self, memory, memory_ahead_mask, memory_padding_mask, tgt, tgt_ahead_mask, tgt_padding_mask):
        """
        :param memory:
        :param memory_ahead_mask:
        :param memory_padding_mask:
        :param tgt:
        :param tgt_ahead_mask:
        :param tgt_padding_mask:
        :return:
        """
        embedded_tgt = self.embedding(tgt)
        positioned_tgt = self.positional_encoding(embedded_tgt)

        output_seq = self.decoder_layers(tgt,
                                         memory,
                                         tgt_ahead_mask,
                                         memory_ahead_mask,
                                         tgt_padding_mask,
                                         memory_padding_mask
                                         )

