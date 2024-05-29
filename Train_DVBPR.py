# %%
## Loaded all the Libraries ##
import os
import json
import numpy as np
import torch
from torch import nn
import torchvision
import random
from tqdm import *
from PIL import Image
from io import StringIO, BytesIO
import lpips

import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader


from pro_gan_pytorch.networks import create_generator_from_saved_model
from pro_gan_pytorch.utils import adjust_dynamic_range
from torch.nn.functional import interpolate
from train_log import MeanTracker

from latent_shift_predictor import LatentReconstructor
from torch.utils.tensorboard import SummaryWriter

# %%
device = 1
seed = 2

data_train = 'amazon'
out_dir = '../PROGAN_AM_Fashion/my_implementation_rec_mean_var'
latent_dim = 512
learning_rate = 0.0002
training_epoch = 100
batch_size = 64
numofworkers= 4
gan_weights= '../PROGAN_AM_Fashion/Model_log_base/models/depth_7_epoch_50.bin' 

torch.cuda.set_device(device)
print('Cuda is Available: ', torch.cuda.is_available())
random.seed(seed)
torch.random.manual_seed(seed)

print('Seed val fized to: ', seed)

# %%
tb_dir = os.path.join(out_dir, 'tensorboard')
models_dir = os.path.join(out_dir, 'models')
os.makedirs(tb_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

checkpoint = os.path.join(out_dir, 'checkpoint.pt')
writer = SummaryWriter(tb_dir)

# %%
if data_train == 'amazon':

    dataset_name = 'AmazonFashion6ImgPartitioned.npy'
    dataset = np.load('../DVBPR/dataset/'+ dataset_name, encoding='bytes', allow_pickle=True)
    [_, _, _, Item, _, itemnum] = dataset

elif data_train == 'tradesy':

    dataset_name = 'TradesyImgPartitioned.npy'
    dataset = np.load('../DVBPR/data/' + dataset_name, encoding='bytes')
    [user_train, user_validation, user_test, Item, usernum, itemnum] = dataset
    cold_list = np.load('../data/tradesy_one_k_cold.npy')

# %%
def default_loader(path):
    img_pil =  Image.open(BytesIO(path)).convert('RGB')
    img_tensor = input_transform(img_pil)
    return img_tensor

input_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225])
    # transforms.Normalize((0.6949, 0.6748, 0.6676), (0.3102, 0.3220, 0.3252))])
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

class trainset(Dataset):
    def __init__(self, loader=default_loader):
        self.images_i = file_train_i
        self.loader = loader
        self.class_hot = class_train_i

    def __getitem__(self, index):
        fn_i = self.images_i[index]
        img_i = self.loader(fn_i)
        class_one_hot = torch.from_numpy((self.class_hot[index]).astype(float))
        return img_i, class_one_hot

    def __len__(self):
        return len(self.images_i)



# helper scale function
def scale(x):
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    x = (x - x.min()) / (x.max() - x.min())
    return x

def scale_percept(x):
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    x = (x - x.min()) / (x.max() - x.min())
    x = (x*2)-1
    return x


##### Initialize Dataset Object ######

file_train_i = [Item[i][b'imgs'] for i in range(itemnum)]
class_train_i = [Item[i][b'c'] for i in range(itemnum)]
train_dataset = trainset()
data_loader_train = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers = numofworkers, drop_last=True)
print("Total number of Train Samples: ",len(train_dataset))


# %%
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

latent_reconstructor = LatentReconstructor().cuda()
print("Initialized weights")
latent_reconstructor.apply(weights_init)

class Progan_gen(nn.Module):
    def __init__(self, weights_root):
        super(Progan_gen, self).__init__()
        self.G = create_generator_from_saved_model(weights_root)
        self.G_curr_depth = 7
        self.Actual_G_depth = 8

    def forward(self, z):
        out = self.G(z, self.G_curr_depth)
        return out

    def gen_shifted(self, z, shift):
        return self.forward(z + shift)

G = Progan_gen(gan_weights).cuda()

print('Generator Loaded!!')


criterion = nn.MSELoss()
percept = lpips.LPIPS(net='vgg').cuda()

latent_reconstructor_opt = torch.optim.Adam(
    latent_reconstructor.parameters(), lr=learning_rate)


# %%
for epoch in tqdm(range(training_epoch)):

    latent_reconstructor.train()
    G.eval()
    
    batch_cnt = 0
    rec_loss_epoch = 0
    fixed_real = None


    for batch_i, (real_i, target_i) in enumerate(data_loader_train):
        batch_cnt +=1
        real_i = real_i.cuda()
        target_i = target_i.cuda()

        if fixed_real == None:
            fixed_real = real_i


        
        latent_reconstructor_opt.zero_grad()
        
        gen_latent = latent_reconstructor(real_i)
        gen_image = G(gen_latent)

        calc_loss = criterion(scale(real_i), scale(gen_image)) + \
                    percept(real_i, scale_percept(gen_image)).mean()

        calc_loss.backward()
        latent_reconstructor_opt.step()

        rec_loss_epoch += calc_loss.item()

        writer.add_scalar('Loss/Rec_Loss', calc_loss.item(), batch_i)

    writer.add_scalar('Loss_epoch/Rec_Loss', rec_loss_epoch/batch_cnt, epoch)

    latent_reconstructor.eval()

    with torch.no_grad():
        gen_latent = latent_reconstructor(fixed_real)
        gen_image = G(gen_latent).cpu().detach()
        gen_image = scale(gen_image)
        Fake_images_grid = torchvision.utils.make_grid(gen_image, nrow=8)
        Real_image_grid = torchvision.utils.make_grid(scale(fixed_real.cpu().detach()), nrow=8)

    latent_reconstructor.train()
    
    writer.add_scalar('Loss_epoch_vis/Rec_Loss_vis', criterion(scale(fixed_real.cpu().detach()), gen_image).item(), epoch)

    writer.add_image('epoch/Generated_images', Fake_images_grid, epoch)
    writer.add_image('epoch/Real_images', Real_image_grid, epoch)

    torch.save({
        'epoch': epoch,
        'state_dict': latent_reconstructor.state_dict(),
        'optimizer_state_dict': latent_reconstructor_opt.state_dict(),
        }, checkpoint)



# %%



