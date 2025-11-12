import torch
import clip.clip as clip
import os

ckpt_path = os.path.join(os.path.dirname(__file__), "../ckpt/test/DTD.pth")

(model,_,_) = clip.load("ViT-B/16", jit=False)

checkpoint = torch.load(ckpt_path,map_location=torch.device('cpu'))


model.load_state_dict(checkpoint["state_dict"])
print(checkpoint["iteration"])