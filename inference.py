import torch
from models.UltraLight_VM_UNet import UltraLight_VM_UNet
import numpy as np
from torchvision import transforms
from PIL import Image
import cv2

device = torch.device("cuda")
img = cv2.imread("test.jpg")  # input test img
resume_model = "checkpoints/MambaUnet_best.pth"  # model
h, w = img.shape[:2]
copy_img = img.copy()

target_size = max(h, w)
if max(h, w) % 32 != 0:
   target_size = (max(h, w) // 32 + 1) * 32
transform_data = transforms.Compose([
                                     transforms.ToTensor()])
img = cv2.copyMakeBorder(img, 0, target_size-h, 0, target_size-w, cv2.BORDER_CONSTANT, 0)
img = Image.fromarray(img)
img = transform_data(img).unsqueeze(0)
img = img.to(device)

model = UltraLight_VM_UNet(1, 3).cuda()
model.load_state_dict(torch.load(resume_model))  # if best.pth
# model.load_state_dict(torch.load(resume_model)["model_state_dict"])

for param in model.parameters():
    param.requires_grad = False
model.eval()

out = model(img)
out = out.squeeze(1).cpu().detach().numpy()
out = np.transpose(out, (1, 2, 0))
out = out[:h, :w]
pred = np.where(out>0.5, 0, 1)




