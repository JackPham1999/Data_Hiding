import math
import torch
import torch.nn
import torch.optim
import torchvision
import numpy as np
from model import *
import config as c
import datasets
import modules.Unet_common as common
from torch.utils.data import Dataset, DataLoader
from datasets import Hinet_Dataset
import torchvision.transforms as T
import argparse

from PIL import Image
import torchvision.transforms as transforms
from matplotlib.pyplot import imread


print(torch.__version__)
print(torch.zeros(1).cuda())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser(description='input secret and cover images')
parser.add_argument('--secret', type=str,
                    help='an integer for the accumulator')
parser.add_argument('--cover', type=str,
                    help='an integer for the accumulator')

args = parser.parse_args()



def load(net, optim, name):
    state_dicts = torch.load(name)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
        return net, optim
    except:
        print('Cannot load optimizer for some reason or other')


def gauss_noise(shape):

    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)

def load_model():
    net = Model()
    net.cuda()
    init_model(net)
    net = torch.nn.DataParallel(net, device_ids=c.device_ids)
    params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
    optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)
    print(c.MODEL_PATH + c.suffix)
    net, optim = load(net, optim, c.MODEL_PATH + c.suffix)

    net.eval()
    return net, optim


def encode(net, optim, cover_path, secret_path):

    dwt = common.DWT()
    iwt = common.IWT()


    secret_img = secret_path.replace('.png','').split('/')[-1]
    cover_img = cover_path.replace('.png','').split('/')[-1]

    transform_val = T.Compose([
        T.CenterCrop(c.cropsize_val),
        T.ToTensor(),
    ])

    demoloader = DataLoader(
        Hinet_Dataset(transforms_=transform_val, mode="demo", files=[secret_path, cover_path]),
        batch_size=c.batchsize_val,
        shuffle=False,
        pin_memory=True,
        num_workers=0,  # 1
        drop_last=True
    )
    print(demoloader)

    with torch.no_grad():
        for i, data in enumerate(demoloader):
            data = data.to(device)
            cover = data[data.shape[0] // 2:, :, :, :]
            secret = data[:data.shape[0] // 2, :, :, :]
            cover_input = dwt(cover)
            secret_input = dwt(secret)
            input_img = torch.cat((cover_input, secret_input), 1)

            #################
            #    forward:   #
            #################
            output = net(input_img)
            output_steg = output.narrow(1, 0, 4 * c.channels_in)
            output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
            steg_img = iwt(output_steg)
            torchvision.utils.save_image(steg_img, c.IMAGE_PATH_DEMO_steg + f'{cover_img}_steg_api.png')
            # backward_z = gauss_noise(output_z.shape)
def decode(net, optim, steg_path):
    net, optim = load_model()

    dwt = common.DWT()
    iwt = common.IWT()
    # steg_path = c.IMAGE_PATH_DEMO_steg + f'1-cover_steg_3.png'
    with torch.no_grad():
        image = Image.open(steg_path)

        transform = transforms.ToTensor()
        image_tensor = transform(image)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        #################
        #   backward:   #
        #################
        image_tensor = dwt(image_tensor)
        backward_z_temp = gauss_noise((1, 12, 512, 512))
        output_rev = torch.cat((image_tensor.to(device), backward_z_temp), 1)
        bacward_img = net(output_rev, rev=True)
        secret_rev = bacward_img.narrow(1, 4 * c.channels_in, bacward_img.shape[1] - 4 * c.channels_in)
        secret_rev = iwt(secret_rev)
        # cover_rev = bacward_img.narrow(1, 0, 4 * c.channels_in)
        # cover_rev = iwt(cover_rev)
        torchvision.utils.save_image(secret_rev, c.IMAGE_PATH_DEMO_API + f'secret_recover.png')

# #encode()
# decode()

