import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    d_model: int = 4096 # num_head * head_dim
    n_layers: int = 32
    n_heads: int = 32 # number of heads for queries
    n_kv_heads: Optional[int] = None # number of heads for K and V
    vocab_size: int = 256
    multiple_of: int = 256
    d_ff: Optional[int] = None # Upsampling part of feed forward layer
    norm_eps: float = 1e-5
    dropout: float = 0.1

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048
    flash_attention = True


    # Feedforward
    hidden_dim = None

    device: str = None

def RoPE(head_dim:int, seq_len: int, device: str, theta: float = 10000.0):
    """
    Compute the rotary positional embedding
    :param head_dim: dimension of each head
    :param seq_len: max_seq_len
    :param device: str
    :param theta: A constant that comes from the paper
    :return: (max_seq_len, head_dim // 2)
    """
    # As written in the paper
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameter
    # According to the formula: theta_i = 10000^(-2(i-1)/head_dim) for i=[1, 2, d/2]
    # Shape: (head_dim // 2, )
    exponent = torch.arange(0, head_dim, 2).float()
    # Shape: (head_dim // 2,)
    div_term = 1.0 / theta ** (exponent / head_dim).to(device)

    # Build the pos: (max_seq_len, )
    pos = torch.arange(seq_len, device=device).unsqueeze(1)

    # Broadcast to (max_seq_len, head_dim // 2)
    freqs = pos * div_term

    # Compute the complex number in polar form c = R * exp(i * pos * theta) where R = 1 as follows:
    # Shape (max_seq_len, head_dim // 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_complex



def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    """
    Apply rotation to Q and K matrices, not V since only Q and K is involved in the inner product
    :param x: (batch_size, seq_len, num_head, head_dim)
    :param freqs_complex: (max_seq_len, head_dim // 2)
    :param device: str
    :return:
    """
    # Two number as a group to be applied the rotation matrix
    # Shape: (batch_size, seq_len, num_head, head_dim // 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Shape: (1, seq_len, 1, head_dim // 2), which means to apply rotation to all the samples in the batch
    # and to each head independently
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # Apply the rotation: (batch_size, seq_len, num_head, head_dim // 2)
    x_rotated = x_complex * freqs_complex

    # Transform it into original shape (batch_size, seq_len, num_head, head_dim // 2, 2) -> (batch_size, seq_len, num_head, head_dim)
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)

    return x_out


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    # 定义函数 repeat_kv，接受两个参数：张量 x 和重复次数 n_rep
    # x 是一个形状为 (batch_size, seq_len, n_kv_heads, head_dim) 的张量，分别代表：
    # n_kv_heads: KV 头的数量 (number of key-value heads)
    # head_dim: 每个头的维度大小 (dimension size of each head)
    # n_rep: 重复次数

    # 获取张量的形状 (bs: 批次大小, slen: 序列长度, n_kv_heads: KV 头的数量, head_dim: 每个头的维度)
    batch_size, seq_len, n_kv_heads, head_dim = x.shape

    # 如果 n_rep 为 1，表示不需要重复，直接返回原始张量
    if n_rep == 1:
        return x

    # 执行以下操作以重复 KV 头：
    # 1. 在第 4 维度 (即 None) 上扩展 x，使其形状为 (bs, slen, n_kv_heads, 1, head_dim)
    # 2. 使用 expand 函数将第 4 维度扩展为 n_rep，得到形状 (bs, slen, n_kv_heads, n_rep, head_dim)
    # 3. 最后通过 reshape 将形状重新调整为 (bs, slen, n_kv_heads * n_rep, head_dim)
    # 这会将每个 KV 头重复 n_rep 次
    return (
        x[:, :, :, None, :]                       # 扩展张量，在 n_kv_heads 后增加一个维度
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)  # 扩展 n_rep 次
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)  # 调整形状为新的维度
    )



class RMSNorm(nn.Module):
    # 初始化函数，接受参数：
    # dim: 归一化的维度大小
    # eps: 防止除零的非常小的数值
    def __init__(self, dim: int, eps: float=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        """
        Since RMS is applied before grouped multi-headed
        :param x: (batch_size, seq_len, d_model)
        :return: (batch_size, seq_len, d_model)
        """
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        """

        :param x: (batch_size, seq_len, d_model)
        :return: (batch_size, seq_len, d_model)
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        # If n_kv_heads not specified, fall back and choose multi-headed attention
        # 本质上如果n_kv_heads设置为2, 就说明我们的k, v仅仅使用两个头的数据量，那么(q1, q2) 公用 (k1, v1, k2, v2), (q3, q4)也公用 (k1, v1, k2, v2)
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads

        assert args.n_heads % self.n_kv_heads == 0, "Number of heads should be divisible by the number of kv_heads"

        self.n_heads_q = args.n_heads
        # Number of repetition needed for K and V
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.d_model // self.n_heads

        # 设置权重层，当 X 的结构为 (seq_len, d_model)时
        # 常规的Q = Wq @ X、K = Wk @ X、V = Wv @ X 矩阵的结构应该与 X 一致，也是 (seq_len, d_model)
        # 因此常规的 W 应该是 (d_model,d_model)结构
        # 在多头注意力中，W 应该是 (d_model, d_model/n_heads)
        # 在具有kv缓存的情况下，我们是对所有头上的注意力"并行"计算
        # 因此Q的权重应该是(d_model, d_model)
        # K和V的权重应该是(d_model, d_model/n_heads * n_kv_heads)
        self.Wq = nn.Linear(args.d_model, args.n_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(args.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(args.d_model, self.n_kv_heads * self.head_dim, bias=False)

        # 输出层上的O的权重不受影响，是(d_model, d_model)
        self.Wo = nn.Linear(args.n_heads * self.head_dim, args.d_model, bias=False)

        # 设置kv缓存初始值
        self.k_cache, self.v_cache = None, None

        # 设置注意力和残差连接上的dropout层和dropout比例
        self.attn_dropout = nn.Dropout(args.dropout)
        self.residual_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        """
        flash attention, 轻量级GPU适用, 几大特点:
        - 使用大矩阵分块，减少峰值内存消耗，同时能够并行计算
        - sparse attention, 使用dropout, 在不改变attention map分布的情况下减小运算量
        - 提升运行速度
        """
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attention

        # 前瞻掩码
        # 用于QK^T矩阵，需要何其形状保持一致
        # QK^T 矩阵,对于每个heads都有一个 (batch_size, num_heads, seq_len, seq_len)
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)

        # buffer用于保存神经网络中除了权重之外、需要被保存的静态数据们
        # 比如掩码矩阵、比如位置编码中的频率等等编码表
        # "mask"我们指定的buffer名称，我们可以通过self.mask来调出掩码矩阵
        self.register_buffer("mask", mask, persistent=False)

        self.device = args.device

    def forward(self, x: torch.Tensor, kv_cache=False):
        """

        :param x: (batch_size, seq_len, d_model)
        :param freqs_complex: (max_seq_len, head_dim // 2)
        :param kv_cache:
        :return:
        """

        batch_size, seq_len, _ = x.shape
        Q = self.Wq(x)

        if self.train():
            K, V = self.Wk(x), self.Wv(x)

        # 如果是推理模式，且kv_cache设置是打开的
        # 那要判断现在是否是初次预测
        if kv_cache and self.eval():
            if all(cache is not None for cache in (self.k_cache, self.v_cache)):
                # 如果不是None，说明不是初次预测了，此时需要的是缓存更新
                xk_new_token = self.Wk(x[:, -1, :]).unsqueeze(1)
                xv_new_token = self.Wv(x[:, -1, :]).unsqueeze(1)
                K = torch.cat((self.k_cache, xk_new_token), dim=1)
                V = torch.cat((self.v_cache, xv_new_token), dim=1)
            else:
                # 如果k和v缓存中有一个为None，说明是初次预测
                K, V = self.Wk(x), self.Wv(x)
            self.k_cache, self.v_cache = K, V

        Q = Q.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        K = K.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        freqs_complex = RoPE(self.head_dim, seq_len, device=self.device)

        # 在Q和K上执行旋转位置编码
        Q = apply_rotary_embeddings(Q, freqs_complex, device=self.device)
        K = apply_rotary_embeddings(K, freqs_complex, device=self.device)

        # 将k矩阵和v矩阵进行重复
        K = repeat_kv(K, self.n_rep)
        V = repeat_kv(V, self.n_rep)

        # 矩阵乘法计算注意力分数时，要将n_heads作为第二维度
        # 因为实际要进行乘法的应该时 (seqlen, head_dim) 这样的二维表
        # transpose交换维度，结构变为(batch_size, n_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 如果使用flash attention的话
        # 就调用nn.functional下面的点乘注意力计算方法
        if self.flash and seq_len != 1:
            output = torch.nn.functional.scaled_dot_product_attention(Q, K, V
                                                                      , attn_mask=None  # 这里是padding掩码
                                                                      , dropout_p=self.dropout if self.training else 0.0
                                                                      , is_causal=True  # 这里是自动化的前瞻掩码
                                                                      )
        else:
            # 不使用flash attention，就自己计算
            # 这里的transpose是对最后两个维度的转置
            scores = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)

            # 在注意力分数上放上掩码
            # 如果有kv缓存的话，现在我们的kv矩阵可能会比掩码矩阵要大了
            # 获取缓存的长度
            cache_len = self.k_cache.shape[1] if self.k_cache is not None else 0
            total_len = cache_len + 1  # 当前总长度，等于历史缓存长度 + 当前序列长度

            # 检查是否需要扩展掩码矩阵
            if total_len > self.mask.shape[-1]:
                # 动态生成新的掩码，大小为 (seq_len + cache_len, seq_len + cache_len)
                new_mask = torch.full((1, 1, total_len, total_len), float("-inf")).to(x.device)
                new_mask = torch.triu(new_mask, diagonal=1)  # 生成前瞻掩码
                self.mask = new_mask  # 更新掩码矩阵

            scores = scores + self.mask[:, :, :seq_len, :seq_len]

            # 对最后一个维度求解softmax
            scores = F.softmax(scores.float(), dim=-1).type_as(Q)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, V)  # (bs, n_local_heads, seqlen, head_dim)

            # 最后再将结构转回来，并且将n_heads中的所有信息合并
            # contiguous() 用于确保张量在内存中的存储是连续的
            # 特别是在经过某些操作（如 transpose）后，这对后续的 view() 等操作至关重要，以避免错误
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 注意力机制的输出
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.d_model = args.d_model
        self.head_dim = self.d_model // self.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(
            dim=args.d_model,  # 模型的总维度
            hidden_dim=args.hidden_dim,  # 前馈网络隐藏层的维度
            multiple_of=args.multiple_of,  # 前馈网络隐藏层的维度应是此数的倍数
            dropout=args.dropout,  # dropout 概率
        )

        # Normalization before self-attention
        self.attention_norm = RMSNorm(args.d_model, eps=args.norm_eps)

        # Normalization before the feedforward block
        self.ffn_norm = RMSNorm(args.d_model, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos:int, freqs_complex: torch.Tensor):
        """
        :param x: (batch_size, seq_len, d_model)
        :param start_pos: int
        :param freqs_complex: (max_seq_len, head_dim // 2)
        :return: (batch_size, seq_len, d_model)
        """

        # Shape (batch_size, seq_len, d_model)
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(x))
        return out



# All layers of LLama model according to the paper, except for the softmax layer
class LLama(nn.Module):

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.embedding = nn.Embedding(self.vocab_size, args.d_model)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))

        self.lmsNorm = RMSNorm(args.d_model, eps=args.norm_eps)
        self.fc_out = nn.Linear(args.d_model, self.vocab_size, bias=False)

        self.rope = RoPE(args.d_model // self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device)

    def forward(self, x: torch.Tensor, start_pos: int):
        """
        The forward pass, for decoder-only transformer like Llama
        :param x: (batch_size, seq_len), seq_len may vary across batches
        :param start_pos: used for KV cache to notify which part should we use KV cache
        :return:
        """
        batch_size, seq_len = x.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        x = self.embedding(x) # (batch_size, seq_len, d_model)

        freqs_complex = self.rope[start_pos: start_pos + seq_len]

        for layer in self.layers:
            x = layer(x, start_pos, freqs_complex)

        x = self.lmsNorm(x)
        x = self.fc_out(x)

        return x
