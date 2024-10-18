from .common_import import *
def create_padding_mask(seq, pad_token=0):
    """
    当输入的src是(batch_size, seq_len)的未经过embedding的形态时,
    比如: [[2,3,6,5,8,4,0,0],
        [5,2,6,3,3,33,3,0],
        [965,434,56,32,0,0,0,0]]
    :param seq: (batch_size, seq_len)
    :param pad_token:
    :return: src_padding_mask (batch_size, seq_len)
    按照上面的输入，我们的输出就是
    [[0,0,0,0,0,0,1,1],
     [0,0,0,0,0,0,0,1],
     [0,0,0,0,1,1,1,1]]
    """
    src_padding_mask = (seq == pad_token)
    src_padding_mask = src_padding_mask.float() * -1e09
    return src_padding_mask


def create_padding_mask_embedded(seq, pad_token=0):
    """
    当输入的src是(batch_size, seq_len, d_model)的经过embedding的形态时,
    比如: [
            [[-0.23, -0.89, -0.1, 0.8, 0.3, 0, 0.2],
            [0.13, 0.99, 0.51, 0.28, 0.23, 0.001, 0.02],
            [-0.23, -0.89, -0.1, 0.8, 0.3, 0, 0.2],
            [0, 0, 0, 0, 0, 0, 0]],

            [[-0.23, -0.89, -0.1, 0.8, 0.3, 0, 0.2],
            [0.13, 0.99, 0.51, 0.28, 0.23, 0.001, 0.02],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0]]
          ]
    由于对token的embedding的结果中可能含有0，此时只有全零的行才能被认为是被padding的token
    :param seq:
    :param pad_token:
    :return: src_padding_mask (batch_size, seq_len)
    按照上面的输入，我们的输出就是
    [[0,0,0,1],
     [0,0,1,1]]
    """
    src_padding_mask = (seq == pad_token).all(dim=-1)
    src_padding_mask = src_padding_mask.float() * -1e09
    return src_padding_mask



def create_ahead_mask(seq_len, start_seq=1):
    """
    :param seq_len:
    :param start_seq:
    :return:
    """
    mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=start_seq)  # 上三角矩阵
    mask = mask.float() * -1e9  # 将未来的位置设置为负无穷大
    return mask  # (seq_len, seq_len)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512, batch_first=True):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = pe.unsqueeze(0)
        else:
            pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
