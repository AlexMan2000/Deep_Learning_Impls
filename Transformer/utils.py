from common_import import *
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
    """
    - 位置编码完全没有使用我们embedding层输出的具体值，而仅仅利用了输出的形状，所以位置编码与语义无关
    - 位置编码层没有任何需要学习的参数，于是用不到nn.Parameter
    - max_seq_len旨在帮助我们的位置编码快速预先计算好所有的正余弦函数值，让代码执行速度更快
    """
    def __init__(self, d_model, dropout, max_seq_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        positional_matrix = torch.zeros((max_seq_len, d_model))

        position = torch.arange(0, max_seq_len).unsqueeze(1)  # (max_seq_len, 1), waiting to be broadcasted
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        ) # (d_model // 2 + 1, )

        # position * div_term will be (max_seq_len, d_model)
        # 预先计算好max_seq_len个位置的编码，在前向传播的过程中仅需要索引即可，速度是很快的
        positional_matrix[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            positional_matrix[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            positional_matrix[:, 1::2] = torch.cos(position * div_term)

        # Used for those parameters that are fixed/not learnable, but are used across training and testing
        # Used for faster retrieval
        self.register_buffer("pe", positional_matrix)

    def forward(self, x):
        """
        Input should be the embedding src token sequence
        :param x: (batch_size, seq_len, d_model)
        :return: dropout(x + pe)
        """
        pe = self.pe.unsqueeze(0) # (1, seq_len, d_model), broadcast it across all samples in the batch

        # This disabling grad is optional since pe is not a learnable parameter by definition
        x = x + pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


