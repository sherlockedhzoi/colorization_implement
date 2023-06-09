# import

import torch as tch
import torch.nn as nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from nets import *

# load dataset

batch_size=10
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307, ),(0.3081, ))])

train_ds=datasets.CIFAR10(root='./dataset/CIFAR10',train=True,download=True,transform=transform)
train_dsloader=DataLoader(train_ds, batch_size=batch_size, shuffle=True)

test_ds=datasets.CIFAR10(root="./dataset/CIFAR10",train=False,download=True,transform=transform)
test_dsloader=DataLoader(test_ds,batch_size=batch_size,shuffle=False)

# get CNN

model=[eccv16(),siggraph17()]
test_num=0

# loss and optimizer
loss_func=nn.MSELoss()
optimizer=optim.SGD(model[test_num].parameters(),lr=0.01,momentum=0.5)

# train
def train(epoch):
    running_loss=0.
    for batch_id,data in enumerate(train_dsloader,0):
        imgs,tags=data
        inputs,targets=preprocess_data(imgs)
        optimizer.zero_grad()

        outputs=model[test_num](inputs)
        loss=loss_func(outputs,targets)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        if batch_id%300==299:
            print("[%d,%5d] loss: %.3f" %(epoch+1,batch_id+1,running_loss/2000))
            running_loss=0.

# test

def test():
    correct=0
    total=0
    with tch.no_grad():
        for data in test_dsloader:
            imgs,tags=data
            inputs,targets=preprocess_data(imgs)

            outputs=model[test_num](inputs)
            predicted=outputs

            total+=target.size(0)
            correct+=(predicted==target).sum().item()
    print("Accuracy on test set: %d %% [%d/%d]"%(100*correct/total,correct,total))
    return 1000*correct/total

# main

if __name__=='__main__':
    len=10
    x=[0]*len
    y=[0]*len
    print(6)
    for epoch in range(10):
        train(epoch)

# TODO the test part
    
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if(opt.use_gpu):
	colorizer_eccv16.cuda()
	colorizer_siggraph17.cuda()

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
if(opt.use_gpu):
	tens_l_rs = tens_l_rs.cuda()

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

plt.imsave('%s_eccv16.png'%opt.save_prefix, out_img_eccv16)
plt.imsave('%s_siggraph17.png'%opt.save_prefix, out_img_siggraph17)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(img_bw)
plt.title('Input')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(out_img_eccv16)
plt.title('Output (ECCV 16)')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(out_img_siggraph17)
plt.title('Output (SIGGRAPH 17)')
plt.axis('off')
plt.show()
