import argparse
import torch
from Transformer.model.TransformerModel import PositionalEncoding
def test_positional_encoding():
    position = PositionalEncoding(d_model=10, dropout=0.5)
    pos_1 = torch.tensor([8.4147e-01, 5.4030e-01, 1.5783e-01, 9.8747e-01, 2.5116e-02,
                          9.9968e-01, 3.9811e-03, 9.9999e-01, 6.3096e-04, 1.0000e+00])
    pos_2 = torch.tensor([9.0930e-01, -4.1615e-01, 3.1170e-01, 9.5018e-01, 5.0217e-02,
                          9.9874e-01, 7.9621e-03, 9.9997e-01, 1.2619e-03, 1.0000e+00])


    assert torch.allclose(pos_1, position.pe[1, :], atol=1e-02, rtol=1e-02) and torch.allclose(pos_2, position.pe[2, :],
                                                                                               atol=1e-02,
                                                                                               rtol=1e-02), print(
        "Incorrect values in positional embeddings")
    print("=" * 10 + "   Positional Encoding Unit Test 1 Passed   " + "=" * 10)

    position = PositionalEncoding(d_model=9, dropout=0.5)
    # pos_1 = torch.tensor([0.84147, 0.54030, 0.15783, 0.98747, 0.02512, 0.99968, 0.00398, 0.99999, 0.00063])
    # pos_2 = torch.tensor([0.90930, -0.41615, 0.31170, 0.95018, 0.05022, 0.99874, 0.00796, 0.99997, 0.00126])

    breakpoint()
    assert torch.allclose(pos_1, position.pe[1, :], atol=1e-02, rtol=1e-02) and torch.allclose(pos_2, position.pe[2, :],
                                                                                               atol=1e-02,
                                                                                               rtol=1e-02), print(
        "Incorrect values in positional embeddings")
    print("=" * 10 + "   Positional Encoding Unit Test 2 Passed   " + "=" * 10)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--positional_encoding", action="store_true", help="unit test for positional encoding")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if args.positional_encoding:
        test_positional_encoding()
