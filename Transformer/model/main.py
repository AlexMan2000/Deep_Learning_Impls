from Transformer.model.TransformerModel import *
from EncoderOnlyTransformer import *
from utils import *

def createData():
    src_data = torch.randint(0, 10, (5, 50))
    tgt_data = torch.randint(0, 15, (5, 50))
    src_seq_lens = [42, 38, 10, 12, 29]
    tgt_seq_lens = [23, 18, 40, 36, 20]
    seq_lens_tensor = torch.tensor(src_seq_lens).unsqueeze(1)  # Shape (5, 1)
    tgt_lens_tensor = torch.tensor(tgt_seq_lens).unsqueeze(1)  # Shape (5, 1)
    src_indices = torch.arange(50).unsqueeze(0)  # Shape (1, 50)
    tgt_indices = torch.arange(50).unsqueeze(0)

    # Create the mask: valid positions (before seq_len) are True, padding positions are False
    src_padding_mask = (src_indices < seq_lens_tensor).float()  # Shape (5, 50)
    tgt_padding_mask = (tgt_indices < tgt_lens_tensor).float()

    return (src_data * src_padding_mask).int(), (tgt_data * tgt_padding_mask).int()


def createEncoderDecoder():
    encoder = TransformerEncoderModel(
        src_vocab_size=10,
        d_model=512,
        num_head=8,
        num_encoder_layers=6,
        d_ff=2048
    )
    decoder = TransformerDecoderModel(
        tgt_vocab_size=15,
        d_model=512,
        num_head=8,
        num_decoder_layers=6,
        d_ff=2048
    )

    model = TransformerModel(encoder, decoder)

    return model


def createEncoderOnlyModel():
    encoderOnlyModel = EncoderOnlyTransformer(
        src_vocab_size=10,
        d_model=512,
        num_head=8,
        num_encoder_layers=6,
        d_ff=2048
    )

    return encoderOnlyModel


if __name__ == "__main__":
    model = createEncoderDecoder()
    encoderModel = createEncoderOnlyModel()
    src_seq, tgt_seq = createData()
    src_padding_mask = create_padding_mask(src_seq, 0)
    tgt_padding_mask = create_padding_mask(tgt_seq, 0)
    tgt_ahead_mask = create_ahead_mask(tgt_seq.size(1))
    # decoder_output = model(
    #     src=src_seq,
    #     tgt=tgt_seq,
    #     src_padding_mask=src_padding_mask,
    #     tgt_ahead_mask=tgt_ahead_mask,
    #     tgt_padding_mask=tgt_padding_mask)
    # print(decoder_output.shape)
    encoder_output = encoderModel(
        src=src_seq,
        src_padding_mask=src_padding_mask
    )
    print(encoder_output.shape)
