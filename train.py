import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import UNet
import MKDataset
from torchvision.utils import save_image

path = r'd:\DRIVE\training'
module = r'weight\net.pth'
weight_save="weight_save"
img_save_path = r'd:\drive_train_img'
epoch = 1

net = UNet.MainNet().cuda()
optimizer = torch.optim.Adam(net.parameters())
lr_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=5,T_mult=2)
loss_func = nn.BCELoss()

dataloader = DataLoader(MKDataset.MKDataset(path), batch_size=2, shuffle=True)

if os.path.exists(module):
    net.load_state_dict(torch.load(module))
else:
    print('No Params!')
if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)

while True:
    for i, (xs, ys) in enumerate(dataloader):
        xs = xs.cuda()
        ys = ys.cuda()
        xs_ = net(xs)

        loss = loss_func(xs_, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print('epoch: {},  count: {},  loss: {}'.format(epoch, i, loss))

            torch.save(net.state_dict(),os.path.join(weight_save,f"{epoch}.pth"))
            print('module is saved !')

        x = xs[0]
        x_ = xs_[0]
        y = ys[0]
        # print(y.shape)
        img = torch.stack([x,x_,y],0)
        # print(img.shape)
        # print(img.shape)

        # save_image(img.cpu(), os.path.join(img_save_path,f'{epoch}{i}.png'))
        # print("saved successfully !")
    epoch += 1