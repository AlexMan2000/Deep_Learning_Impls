from common_import import *
from utils import *

class EncoderOnlyTransformer(nn.Module):
    def __init__(self
                 , src_vocab_size
                 , d_model
                 , num_head
                 , num_encoder_layers
                 , d_ff
                 , dropout=0.1):
        super().__init__()
        """ 
        @param src_vocab_size: 词典大小
        @param d_model: hidden_size
        @param num_head: 注意力头数量
        @param num_encoder_layers: EncoderLayer的数量(论文中是6)
        @param d_ff: 前馈网络的上采样维度
        @param dropout: 所有dropout的概率
        """

        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model, dropout)
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # d_model = h * d_k
            nhead=num_head,
            dim_feedforward=d_ff, # upsampling
            # Dropout 概率，默认值为 0.1。在Transformer架构图中虽然没有展现dropout层，
            # 但现在业内习惯于将Dropout层放置在每一个复杂结构之后，
            # 在Encoder中，Dropout出现在自注意力层后、残差链接之前，也出现在前馈神经网络后、残差链接之前
            dropout=dropout,
            activation=F.relu, # default to be relu
            batch_first=True # (batch_size, seq_len, d_model)
        )

        # N x 编码层, 实际上就是做了一个克隆, copy.deepcopy
        self.encoder_layers = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )

        # 输出层, 如果我们需要使用encoder做回归任务
        self.fc_out = nn.Linear(d_model, 1)


        # 用于二分类任务, 比如情感分类
        # self.fc_out = nn.Sequential(
        #         nn.Linear(d_model, 1),
        #         nn.Sigmoid()
        # )

        # 用于多分类任务， 比如
        # self.fc_out = nn.Sequential(
        #         nn.Linear(d_model, num_classes),
        #         nn.Softmax(dim=-1)
        # )

    def forward(self, src, src_mask=None, src_padding_mask=None):
        """
        :param src: 进行embedding之前的张量 (batch_size, seq_len) a list of token_index
        :param src_mask: 前瞻掩码， 一般对于encoder来说不用，(seq_len, seq_len)
        :param src_padding_mask: padding掩码，必须使用，解决句子中的零分布不均的问题 (batch_size, seq_len)
        :return: (batch_size, 1) for regression task
        """

        # Step 1: embedding input
        # (batch_size, seq_len) -> (batch_size, seq_len, d_model)
        # 由于embedding过后的编码向量中的数值都很小，会导致梯度消失，模型迭代效果不佳的现象，于是需要缩放一下
        # 更何况后续在attention运算的过程中会涉及到除以sqrt{d_model}，于是先scale up一下更好
        embedded_src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model))

        # Step 2: Apply Positional Encoding
        positioned_src = self.position_encoding(embedded_src)

        # Step 3: (batch_size, seq_len, d_model) -> encoder layers x N -> (batch_size, seq_len, d_model)
        output_seq = self.encoder_layers(src=positioned_src, mask=src_mask, src_key_padding_mask=src_padding_mask) # (batch_size, seq_len, d_model)


        # Optional
        # Step 4: Output for downstream task (batch_size, seq_len, d_model) -> (batch_size, d_model)
        # Basically just compute mean across all sequences
        output = output_seq.mean(dim=1)
        # dim = 1意味着我们对每个句子中的词做一个平均，适合用于句子级别的情感分类
        # dim = 0意味着我们对不同句子中的同一个位置的单词做一个平均，适合用于单词级别的词性标注

        # Step 5: Output for downstream task, passing through linear layer(s)
        output = self.fc_out(output) # (batch_size, d_model) -> (batch_size, 1)

        return output
