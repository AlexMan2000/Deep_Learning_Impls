from collections import defaultdict

import nltk
import torch
from nltk.corpus import sentence_polarity
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from Transformer.model.EncoderOnlyTransformer import EncoderOnlyTransformer
from Transformer.model.utils import create_padding_mask

nltk.download("sentence_polarity")


class Vocab:

    def __init__(self, tokens=None):
        """
        对于一个文本进行分词构建
        如果传入的是一个分好词的unique token list, 就可以直接使用分词的结果
        如果是原始文本数据，则需要调用类中的build方法对每一句句子进行分词
        :param tokens: A list of unique words in the corpus(multiple sentences)
        """
        self.idx_to_token = list()
        self.token_to_idx = dict()
        self.vocab_size = 0

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = ["<unk>"] + tokens  # unk -> 3
            if "<sos>" not in tokens:
                tokens = ["<sos>"] + tokens  # eos -> 2
            if "<eos>" not in tokens:
                tokens = ["<eos>"] + tokens  # sos -> 1
            if "<pad>" not in tokens:
                tokens = ["<pad>"] + tokens  # pad -> 0
            for token in tokens:
                self.token_to_idx[token] = len(self.idx_to_token)
                self.idx_to_token.append(token)

            self.unk = self.token_to_idx["<unk>"]
            self.pad = self.token_to_idx["<pad>"]
            self.sos = self.token_to_idx["<sos>"]
            self.eos = self.token_to_idx["<eos>"]
            self.vocab_size += len(tokens)
        else:
            raise RuntimeError("Should pass in the list of tokens you are willing to use!")

    @classmethod
    def build(cls,
              corpus,
              min_freq=1,
              stopwords=None,
              preprocessing=False,
              reserved_tokens=None):
        """

        :param cls: Vocab这个类本身，这魔法命令classmethod的要求有了cls就可以在不进行实例化的情况下直接调用build功能
        :param corpus: List[string] 需要构建词汇表和词典的文本，在这个文本上我们可以直接开始进行词频筛选。注意！这个文本的范围很广泛，只要不是单一token list，都可以被认为是文本
        :param min_freq: 用于筛选的最小频率，低于该频率阈值的词会被删除
        :param stopwords:
        :param preprocessing:
        :param reserved_tokens: 我们可以选择性输入的"通用词汇表"，假设text本身太短词太少的话reserved_token可以帮助我们构建更大的词典、从而构建更大的词向量空间
        :return: 一个Vocab类的实例化对象
        """
        if stopwords is None:
            stopwords = {"的", "和", "了", "在", "是", "就", "不", "也", "有", "但"}

        token_freqs = defaultdict()

        # 分词: 可用其他分词库代替
        for sentence in corpus:
            # 这里是按照unicode字符分词
            for token in sentence:
                token_freqs[token] = token_freqs.get(token, 0) + 1

        unique_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        unique_tokens += [token for token, freq in token_freqs.items() if
                          freq >= min_freq and token != "<unk>" and token not in stopwords]
        return cls(unique_tokens)

    def __len__(self):
        """
        Return the length of the vocabulary
        :return:
        """
        return len(self.idx_to_token)

    def __getitem__(self, token):
        """
        Return the index of the given token(string)
        3 if token not in the vocab list, literally unknown
        :param token: string, the token
        :return:
        """
        return self.token_to_idx.get(token, self.unk)

    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_idx[token] for token in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.idx_to_token[index] for index in ids]


def create_dataset():
    # 导入NLTK库中的sentence_polarity模块
    from nltk.corpus import sentence_polarity

    # 创建一个词汇表vocab
    # 词汇表是在字典之前、先对句子中出现的所有不重复的词进行统计的表单
    # 先构建词汇表vocab，再对vocab进行编码就可以构成字典
    vocab = Vocab.build(sentence_polarity.sents())


    # 构建训练数据集
    # 将正面情感的句子标记为0，取前4000个正面句子
    # 负面情感的句子标记为1，取前4000个负面句子
    train_data = [(vocab.convert_tokens_to_ids(sentence), 0)
                  for sentence in sentence_polarity.sents(categories='pos')[:4000]] \
                 + [(vocab.convert_tokens_to_ids(sentence), 1)
                    for sentence in sentence_polarity.sents(categories='neg')[:4000]]

    # 构建测试数据集
    # 使用剩余的正面情感句子，标记为0
    # 使用剩余的负面情感句子，标记为1
    test_data = [(vocab.convert_tokens_to_ids(sentence), 0)
                 for sentence in sentence_polarity.sents(categories='pos')[4000:]] \
                + [(vocab.convert_tokens_to_ids(sentence), 1)
                   for sentence in sentence_polarity.sents(categories='neg')[4000:]]

    # 返回训练数据、测试数据和vocab类本身
    return train_data, test_data, vocab


# 定义TransformerDataset类
# 用于为数据赋予len和getitem属性
# 从而让数据能够适应PyTorch的数据结构
class TransformerDataset(Dataset):
    def __init__(self, data):
        # 初始化数据集，将传入的数据保存在实例变量data中
        self.data = data

    def __len__(self):
        # 返回数据集的大小
        return len(self.data)

    def __getitem__(self, i):
        # 根据索引i获取数据集中的第i个样本
        return self.data[i]


# 定义collate_fn函数，用于在DataLoader中对一个batch的数据进行处理
def collate_fn(examples):
    """

    :param examples: A batch size of examples, but not bundled
    :return:
    """
    # 将每个样本的输入部分转换为张量（转变为Tensor）
    inputs = [torch.tensor(ex[0]) for ex in examples]

    # 将每个样本的目标部分转换为长整型张量
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)

    # pytorch自带的padding工具
    # 对batch内的样本进行padding，使其具有相同长度, pad to the longest， default pad_token is 0
    inputs = pad_sequence(inputs, batch_first=True)

    # 返回处理后的输入、长度和目标 (batch_size, seq_len)
    return inputs, targets


def train(
        model,
        train_dataloader,
        test_dataloader,
        epoch,
        save=True,
        cuda=False,
):
    device = torch.device("cpu")

    if cuda:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = model.to(device)
    model.train()

    loss_fn = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)



    for i in range(1, epoch + 1):
        total_loss_for_epoch = 0
        for data, target in train_dataloader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss_for_epoch += loss.item()


        if i % 5 == 0:

            acc = 0
            model.eval()

            for data, target in test_dataloader:
                data = data.to(device)
                target = target.to(device)
                with torch.no_grad():
                    output = model(data)
                    acc += (output.argmax(dim=-1) == target).sum().item()

            print(f"Acc: {acc / len(test_dataloader):.2f}")

            model.train()

        print(f"Loss for epoch {i}: {total_loss_for_epoch:.2f}")


    if save:
        model.save("../saved_models/encoder_sentiment.pt")

if __name__ == "__main__":
    train_data, test_data, vocab = create_dataset()
    vocab_size = vocab.vocab_size


    train_dataset = TransformerDataset(train_data)
    test_dataset = TransformerDataset(test_data)
    device = torch.device("cuda")

    # Training
    batch_size = 32
    d_model = 128
    num_classes = 2
    num_epoch = 25

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

    model = EncoderOnlyTransformer(src_vocab_size=vocab_size,
                                   d_model=d_model,
                                   num_head=2,
                                   num_encoder_layers=2,
                                   d_ff=256,
                                   dropout=0.1
                                   )

    train(
        model,
        train_dataloader,
        test_dataloader,
        epoch=num_epoch,
        cuda=True,
    )
