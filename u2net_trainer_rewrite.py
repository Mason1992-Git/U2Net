import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from model import U2NET
from model import U2NETP
import MKDataset
from torchvision.utils import save_image

path = r'G:\09-UNET-EyeballSegmentation\train_data'
module = r'G:\09-UNET-EyeballSegmentation\weight\162.pth'
weight_save=r"G:\09-UNET-EyeballSegmentation\weight"
img_save_path = r'G:\09-UNET-EyeballSegmentation\img_save_process'
epoch = 1
model_name = "u2netp"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if(model_name=='u2net'):
    net = U2NET(3, 1).to(device)
elif(model_name=='u2netp'):
    net = U2NETP(3,1).to(device)

optimizer = torch.optim.Adam(net.parameters())
lr_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
loss_func = nn.BCELoss()

dataloader = DataLoader(MKDataset.MKDataset(path), batch_size=16, shuffle=True)

if os.path.exists(module):
    net.load_state_dict(torch.load(module))
    print("load weight successfully!")
else:
    print('No Params!')
if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)

while True:
    for i, (xs, ys) in enumerate(dataloader):
        xs = xs.to(device)
        ys = ys[:,0:1].to(device)
        xs_ = net(xs)
        # print(xs_[0].shape)
        # print(ys.shape)

        loss = loss_func(xs_[0], ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 3 == 0:
            print('epoch: {},  count: {},  loss: {}'.format(epoch, i, loss))

            torch.save(net.state_dict(),os.path.join(weight_save,f"{epoch}.pth"))
            print('module is saved !')

        x = xs[0][0:1]
        # print(x.shape)
        x_ = xs_[0][0]
        # print(x_.shape)
        y = ys[0]
        # print(y.shape)
        # print(y.shape)
        img = torch.stack([x,x_,y],0)
        # print(img.shape)
        # print(img.shape)

        save_image(img.cpu(), os.path.join(img_save_path,f'{epoch}{i}.png'))
        print("saved successfully !")
    epoch += 1