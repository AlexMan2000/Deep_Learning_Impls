import torchvision
import torchvision.transforms as transforms


# 图像识别
fmnist = torchvision.datasets.FashionMNIST(root="./datasets/FMNIST",
                                    train=True,
                                    download=False,
                                    transform=transforms.ToTensor()
                                    )


omnist = torchvision.datasets.Omniglot(root="./datasets/OMNIST",
                                    background=True,
                                    download=False,
                                    transform=transforms.ToTensor()
                                    )


svhn = torchvision.datasets.SVHN(root="./datasets/SVHN",
                                    split="train",
                                    download=False,
                                    transform=transforms.ToTensor()
                                    )



