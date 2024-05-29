import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

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


from pro_gan_pytorch.networks import create_generator_from_saved_model, create_generator_from_saved_model_opt
from pro_gan_pytorch.utils import adjust_dynamic_range
from torch.nn.functional import interpolate
import torchvision.transforms.functional as fn
from train_log import MeanTracker

from torch.utils.tensorboard import SummaryWriter

import model_rs as recsys_models


# %%
device = 1
seed = 0

data_train = 'amazon'
out_dir = 'dummy_rec/'
latent_dim = 512
learning_rate = 0.00002
training_epoch = 1000
batch_size = 1
numofworkers= 4
gan_weights= '../PROGAN_AM_Fashion/Model_log_base/models/depth_7_epoch_50.bin' 

#torch.cuda.set_device(device)
print('Cuda is Available: ', torch.cuda.is_available())
random.seed(seed)
torch.random.manual_seed(seed)

# print('Seed val fixed to: ', seed)


if data_train == 'amazon':

    dataset_name = 'AmazonFashion6ImgPartitioned.npy'
    dataset = np.load('../DVBPR/dataset/'+ dataset_name, encoding='bytes', allow_pickle=True)
    [user_train, _, _, Item, usernum, itemnum] = dataset

elif data_train == 'tradesy':

    dataset_name = 'TradesyImgPartitioned.npy'
    dataset = np.load('../DVBPR/data/' + dataset_name, encoding='bytes')
    [user_train, user_validation, user_test, Item, usernum, itemnum] = dataset
    cold_list = np.load('../data/tradesy_one_k_cold.npy')


def default_loader(path):
    img_pil =  Image.open(BytesIO(path)).convert('RGB')
    img_tensor = input_transform(img_pil)
    return img_tensor

# input_transform = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406],
#     #                     std=[0.229, 0.224, 0.225])
#     # transforms.Normalize((0.6949, 0.6748, 0.6676), (0.3102, 0.3220, 0.3252))])
#     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
#     ])


input_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

class trainset(Dataset):
    def __init__(self, loader=default_loader):
        self.images_i = file_train_i
        self.images_j = file_train_j
        self.target = train_ls
        self.loader = loader

    def __getitem__(self, index):
        fn_i = self.images_i[index]
        img_i = self.loader(fn_i)
        fn_j = self.images_j[index]
        img_j = self.loader(fn_j)
        target = self.target[index]
        return img_i, img_j, target[0], target[1], target[2]

    def __len__(self):
        return len(self.images_i)



# helper scale function
def scale(x):
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    x = (x - x.min()) / (x.max() - x.min())
    return x

def scale_synth(x):
    # scale to feature_range and return scaled x
    x = adjust_dynamic_range(x,drange_in=(-1.0, 1.0), drange_out=(0.0, 1.0))
    return x

def scale_percept(x):
    # scale to feature_range and return scaled x
    x = adjust_dynamic_range(x,drange_in=(-1.0, 1.0), drange_out=(-1.0, 1.0))

    return x

def manual_normalize(x, mean, std):
    mean_ten = torch.Tensor(mean).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).cuda()
    std_ten = torch.Tensor(std).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).cuda()
    x = (x-mean_ten)/std_ten
    return x

def scale_rs(x):
    x = adjust_dynamic_range(x,drange_in=(-1.0, 1.0), drange_out=(0.0, 1.0))
    x = interpolate(x, size=(224, 224), mode='bilinear')
    x = manual_normalize(x, mean=[0.6949, 0.6748, 0.6676], std=[0.3102, 0.3220, 0.3252])
    return x


##### Initialize Dataset Object ######
def sample(user):
    u = random.randrange(usernum)
    numu = len(user[u])
    i = user[u][random.randrange(numu)][b'productid']
    M=set()
    for item in user[u]:
        M.add(item[b'productid'])
    while True:
        j=random.randrange(itemnum)
        if (not j in M): break
    return (u,i,j)

oneiteration = 0
for item in user_train: oneiteration+=len(user_train[item])

train_ls = [list(sample(user_train)) for _ in range(oneiteration)]


file_train_i = [Item[i][b'imgs'] for i in range(itemnum)]
file_train_j = [Item[i][b'imgs'] for i in range(itemnum)]


train_dataset  = trainset()

data_loader_train = DataLoader(train_dataset, batch_size = 10, shuffle=True, num_workers = numofworkers, drop_last=True)
print("Total number of Train Samples: ",len(train_dataset))


# %%
class Progan_gen(nn.Module):
    def __init__(self, weights_root):
        super(Progan_gen, self).__init__()
        # self.G = create_generator_from_saved_model(weights_root)
        self.G = create_generator_from_saved_model_opt(weights_root)

        self.G_curr_depth = 7
        self.Actual_G_depth = 8

    def forward(self, z, delta):
        out = self.G(z, delta, self.G_curr_depth)
        return out

    # def gen_shifted(self, z, shift, delta):
    #     return self.forward(z + shift, [i+shift.unsqueeze(dim=1).unsqueeze(dim=2) for i in delta])

    def gen_shifted(self, z, shift, delta):
        return self.forward(z + shift, delta)


G = Progan_gen(gan_weights).cuda()
G = G.eval()


# class Progan_gen_1(nn.Module):
#     def __init__(self, weights_root):
#         super(Progan_gen_1, self).__init__()
#         # self.G = create_generator_from_saved_model(weights_root)
#         self.G = create_generator_from_saved_model(weights_root)

#         self.G_curr_depth = 7
#         self.Actual_G_depth = 8

#     def forward(self, z):
#         out = self.G(z, self.G_curr_depth)
#         return out

#     # def gen_shifted(self, z, shift, delta):
#     #     return self.forward(z + shift, [i+shift.unsqueeze(dim=1).unsqueeze(dim=2) for i in delta])

#     def gen_shifted(self, z, shift):
#         return self.forward(z + shift)


# G = Progan_gen_1(gan_weights).cuda()
# G = G.eval()

print('Generator Loaded!!')


criterion = nn.MSELoss()
percept = lpips.LPIPS(net='vgg').cuda()



# %%
#### Defining models for GAN disentaglement
from constants import DEFORMATOR_TYPE_DICT
from latent_deformator import LatentDeformator, DeformatorType
from latent_shift_predictor import LatentShiftPredictorV3
from latent_shift_predictor import LatentReconstructor


## Define Static Variable
CE_loss = nn.CrossEntropyLoss()
# deformator_type = 'my_case'
deformator_type = 'my_case'

deformator_random_init = True
shift_predictor_size = None
shift_predictor_type = 'ResNet'
shift_distribution_key = 0 ## 0 for normal || 1 for uniform
seed = 2
device = 2
multi_gpu = False


shift_scale = 6.0
min_shift = 0.5
shift_distribution_type = shift_distribution_key

deformator_lr = 0.01
shift_predictor_lr = 0.01
n_steps = int(1000)
batch_size = 6

directions_count = batch_size
max_latent_dim = 512

label_weight = 1.0
shift_weight = 0.25
print_Every = 100


## takes in a list of [direction_count]

# for all those batches fetch the direction_count index
# generate a shift vector denotes how much you want to shift those directions corresponding target_indices
# manually normalize every shift magnitude above minimum shift
# for each of those batch size images (create a vector that basically tells for which images you will
# continued: modify which target indice and by how much)

def make_shifts(latent_dim):
    target_indices = torch.randperm(
        directions_count)[:batch_size].cuda()
    if shift_distribution_type == 0:
        shifts = torch.randn(target_indices.shape, device='cuda')
    elif shift_distribution_type == 1:
        shifts = 2.0 * torch.rand(target_indices.shape, device='cuda') - 1.0

    shifts = shift_scale * shifts
    shifts[(shifts < min_shift) & (shifts > 0)] = min_shift
    shifts[(shifts > -min_shift) & (shifts < 0)] = -min_shift

    try:
        latent_dim[0]
        latent_dim = list(latent_dim)
    except Exception:
        latent_dim = [latent_dim]

    z_shift = torch.zeros([batch_size] + latent_dim, device='cuda')
    for i, (index, val) in enumerate(zip(target_indices, shifts)):
        z_shift[i][index] += val

    return target_indices, shifts, z_shift



def make_shifts_1(shifts, z):
    """
    shifts shape should be (batch_size, 1)
    z shape should be (batch_size, 512) ProGAN
    """
    target_indices = torch.randperm(
        batch_size)[:batch_size].cuda()

    # shifts = shifts[target_indices]
    shifts = shift_scale * shifts
    shifts[(shifts < min_shift) & (shifts > 0)] = min_shift
    shifts[(shifts > -min_shift) & (shifts < 0)] = -min_shift

    return shifts


def calculate_percept_loss(img_shifted):
    k = img_shifted

    total_loss = 0

    for i in range(len(img_shifted)):
        fix_image = img_shifted[i].unsqueeze(dim=0).repeat(batch_size-1,1,1,1)
        rest_images = torch.cat([img_shifted[0:i], img_shifted[i+1:]], dim=0)
        perceptual_loss = percept(fix_image,rest_images).mean()
        total_loss += perceptual_loss

    return total_loss/len(img_shifted)


def log_train(step, should_print=True, stats=()):
    if should_print:
        out_text = '{}% [step {}]'.format(int(100 * step / n_steps), step)
        for named_value in stats:
            out_text += (' | {}: {:.2f}'.format(*named_value))
        print(out_text)

## for unpref images
#preferred_images, unpref_images, u, _, _ = next(iter(data_loader_train))


### for pref images
#unpref_images, _, _, _, _ = next(iter(data_loader_train))

num_of_users = 10000
u = torch.LongTensor(random.sample(range(0, usernum), num_of_users))

all_user_pref = []
base_user_pref = []
item_user_pref = []
item_images = []
base_gen_images = []
shifted_images = []

# for u_i in u:
for u_i in range(1000):


    if shift_predictor_type == 'ResNet':
        shift_predictor = LatentShiftPredictorV3(
            batch_size, shift_predictor_size).cuda()    
    elif shift_predictor_type == 'LeNet':
        shift_predictor = LeNetShiftPredictor(
            batch_size, 1 if gan_type == 'SN_MNIST' else 3).cuda()

    latent_recon = LatentReconstructor(batch_size).cuda()
    print('Shift-predictor Loaded!')



    print('Working on Recommender Models!!')

    latent_rs_model = 100

    def init_weights_rs(m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)

    rs_model = nn.DataParallel(recsys_models.pthDVBPR(latent_rs_model))
    rs_model.apply(init_weights_rs)

    print('Loading Rs Model Weights!!!!')
    rs_path = '../AIP/models/ckpt/Newamazon_K100_19.tar'
    rs_Weights = torch.load(rs_path)

    rs_model.load_state_dict(rs_Weights['model_state_dict'])
    rs_model.eval()

    thetau = torch.from_numpy(rs_Weights['U']).cuda()

    G.cuda().eval()

    dlatent=torch.randn((1,512),requires_grad=True,device='cuda')
    dlatent_mean, dlatent_std = dlatent.mean(), dlatent.std()
    dlatent = (dlatent-dlatent_mean)/dlatent_std

    dlatent_img = scale_percept(G(dlatent, [0,0]))

    user_nu = u.cuda()
    ### User Preference
    rs_in1 = dlatent_img
    rs_feat1 = rs_model(rs_in1)
    # Preference_score = torch.log(torch.sigmoid(torch.mul(thetau[user_nu],rs_feat).sum(1))).mean()
    Preference_score1 = torch.mul(torch.index_select(thetau, dim=0, index=user_nu),rs_feat1).sum(1).mean()

    dlatent_shift=torch.randn((batch_size,512),requires_grad=True,device='cuda')
    shift_opt = torch.optim.Adam([dlatent_shift], lr=deformator_lr)
    shift_predictor.cuda().train()
    shift_predictor_opt = torch.optim.Adam(
        shift_predictor.parameters(), lr=shift_predictor_lr)

    dlatent = dlatent.detach()
    n_steps = 1000
    for step in range(0, n_steps, 1): 
        G.zero_grad()
        ##### Optimizing dlatent shift #####
        shift_opt.zero_grad()
        shift_predictor_opt.zero_grad()

        target_indices = torch.randperm(batch_size)[:batch_size].cuda()
        max_shift = 0.5
        shifts = max_shift * dlatent_shift[target_indices]
        shifts = torch.clamp(shifts, min=-max_shift, max=max_shift)
        z_shift = dlatent.detach() + shifts


        z_shift_mean, z_shift_std = z_shift.mean(), z_shift.std()
        z_shift = (z_shift-z_shift_mean)/z_shift_std

        # Deformation
        imgs = scale_percept(G(z_shift,[0,0]))
        logits, predicted_shifts = shift_predictor(imgs)
        logit_loss = label_weight * CE_loss(logits, target_indices)
        shift_loss = torch.mean(torch.abs(predicted_shifts - shifts))

        ### User Preference
        rs_in = imgs
        rs_feat = rs_model(rs_in)
        # Preference_score = torch.log(torch.sigmoid(torch.mul(thetau[user_nu],rs_feat).sum(1))).mean()
        Preference_score = torch.mul(torch.index_select(thetau, dim=0, index=user_nu),rs_feat.unsqueeze(dim=1)).mean(1).sum(1)

        # total loss
        loss = 0.1*logit_loss + shift_loss + (-1*Preference_score.mean())
        # loss = -1*Preference_score
        loss.backward()

        shift_opt.step()

        shift_predictor_opt.step()
        # update statistics trackers
        if step%100==0:
            print("#user: {} iter{}: loss: {:.2f}| logit_loss: {:.2f}| shift_loss: {:.2f}| Preference_score: {:.2f}".format(len(user_nu), step,loss.item(),logit_loss.item(),shift_loss.item(),Preference_score.mean().item()))

    user_nu_preference = Preference_score.cpu().detach().unsqueeze(dim=0)
    all_user_pref.append(user_nu_preference)

    user_base_preference = Preference_score1.cpu().detach().unsqueeze(dim=0)
    base_user_pref.append(user_base_preference)

    base_gen_images.append(dlatent_img.detach().cpu())
    shifted_images.append(imgs.detach().cpu())


all_user_pref = torch.cat(all_user_pref, dim=0)
base_user_pref = torch.cat(base_user_pref, dim=0)
base_gen_images = torch.cat(base_gen_images, dim=0)
shifted_images = torch.cat(shifted_images, dim=0)


torch.save(all_user_pref, 'raw_data_exp/multi_user/preference_score_U_{}.pt'.format(num_of_users))
torch.save(base_user_pref, 'raw_data_exp/multi_user/base_preference_score_U_{}.pt'.format(num_of_users))
torch.save(base_gen_images, 'raw_data_exp/multi_user/base_images_U_{}.pt'.format(num_of_users))
torch.save(shifted_images, 'raw_data_exp/multi_user/shifted_images_U_{}.pt'.format(num_of_users))




print(all_user_pref.shape)
print(base_user_pref.shape)
print(base_gen_images.shape)
print(shifted_images.shape)


print('Done!!!')

