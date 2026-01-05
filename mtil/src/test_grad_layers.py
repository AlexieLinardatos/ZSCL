import torch


dictionary = torch.load("./ckpt/DTD_orthogonal_test4/grad_DTD.pth")

for name, gradient in dictionary.items():
    print(f"{name}: {gradient.shape}")