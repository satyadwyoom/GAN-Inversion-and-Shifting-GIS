import os
import json
import numpy as np
import torch
from torch import nn
import torchvision
import random

from pro_gan_pytorch.networks import create_generator_from_saved_model
from pro_gan_pytorch.utils import adjust_dynamic_range
from torch.nn.functional import interpolate
from train_log import MeanTracker

from latent_deformator import LatentDeformator, DeformatorType
from latent_shift_predictor import LatentShiftPredictor, LeNetShiftPredictor
from constants import DEFORMATOR_TYPE_DICT
from torch.utils.tensorboard import SummaryWriter


## Define Static Variable
out_dir = '../PROGAN_AM_Fashion/my_implementation'
gan_weights = '../PROGAN_AM_Fashion/Model_log_base/models/depth_7_epoch_50.bin'
deformator_type = 'linear'
deformator_random_init = True
shift_predictor_size = None
shift_predictor_type = 'ResNet'
shift_distribution_key = 0 ## 0 for normal || 1 for uniform
seed = 2
device = 1
multi_gpu = False


shift_scale = 6.0
min_shift = 0.5
shift_distribution_type = shift_distribution_key

deformator_lr = 0.0001
shift_predictor_lr = 0.0001
n_steps = int(1e+5)
batch_size = 32

directions_count = 200
max_latent_dim = 512

label_weight = 1.0
shift_weight = 0.25

steps_per_log = 1000
steps_per_save = 10000
steps_per_img_log = 1000
steps_per_backup = 1000

truncation = None

torch.cuda.set_device(device)
print('Cuda is Available: ', torch.cuda.is_available())
random.seed(seed)
torch.random.manual_seed(seed)

print('Seed val fized to: ', seed)


# %%
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

G = Progan_gen(gan_weights)

print('Generator Loaded!!')

# %%
## deformator takes in the shape (2,200,1,1)
deformator = LatentDeformator(shift_dim=[max_latent_dim],
                                input_dim=directions_count,
                                out_dim=max_latent_dim,
                                type=DEFORMATOR_TYPE_DICT[deformator_type],
                                random_init=deformator_random_init).cuda()

## shift predictor takes in a two set of images (real and shifted)
## output is (logits value, magnitude of shift[which image in batch was shifted by how much])
if shift_predictor_type == 'ResNet':
    shift_predictor = LatentShiftPredictor(
        deformator.input_dim, shift_predictor_size).cuda()
elif shift_predictor_type == 'LeNet':
    shift_predictor = LeNetShiftPredictor(
        deformator.input_dim, 1 if gan_type == 'SN_MNIST' else 3).cuda()

print('Deformator & Shift-predictor Loaded!')


# %%
### make various dirs ###

log_dir = os.path.join(out_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
CE_loss = nn.CrossEntropyLoss()

tb_dir = os.path.join(out_dir, 'tensorboard')
models_dir = os.path.join(out_dir, 'models')
images_dir = os.path.join(log_dir, 'images')
os.makedirs(tb_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

checkpoint = os.path.join(out_dir, 'checkpoint.pt')
writer = SummaryWriter(tb_dir)
out_json = os.path.join(log_dir, 'stat.json')
fixed_test_noise = None

# %%
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

# %%
@torch.no_grad()
def validate_g_img(G, deformator):
    deformator.eval()

    z = torch.randn([1] + [max_latent_dim]).cuda()
    imgs = [G(z).detach().cpu()]

    target_indices = torch.arange(
        0, directions_count)[:batch_size].cuda()
    if shift_distribution_type == 0:
        shifts = torch.randn(target_indices.shape, device='cuda')
    elif shift_distribution_type == 1:
        shifts = 2.0 * torch.rand(target_indices.shape, device='cuda') - 1.0

    shifts = shift_scale * shifts
    shifts[(shifts < min_shift) & (shifts > 0)] = min_shift
    shifts[(shifts > -min_shift) & (shifts < 0)] = -min_shift

    z_shift = torch.zeros([batch_size] + [deformator.input_dim], device='cuda')
    for i, (index, val) in enumerate(zip(target_indices, shifts)):
        z_shift[i][index] += val

    imgs_shifted = G.gen_shifted(z, deformator(z_shift))

    imgs.append(imgs_shifted.detach().cpu())

    samples = torch.cat(imgs, dim=0)

    scale_factor = int(2**(8-7))
    if scale_factor > 1:
        samples = interpolate(samples, scale_factor=scale_factor)

    samples = adjust_dynamic_range(
        samples, drange_in=(-1.0, 1.0), drange_out=(0.0, 1.0)
    )

    return samples

# %%
def start_from_checkpoint(deformator, shift_predictor):
    step = 0
    if os.path.isfile(checkpoint):
        state_dict = torch.load(checkpoint)
        step = state_dict['step']
        deformator.load_state_dict(state_dict['deformator'])
        shift_predictor.load_state_dict(state_dict['shift_predictor'])
        print('starting from step {}'.format(step))
    return step

def log_train(step, should_print=True, stats=()):
    if should_print:
        out_text = '{}% [step {}]'.format(int(100 * step / n_steps), step)
        for named_value in stats:
            out_text += (' | {}: {:.2f}'.format(*named_value))
        print(out_text)
    for named_value in stats:
        writer.add_scalar(named_value[0], named_value[1], step)

    with open(out_json, 'w') as out:
        stat_dict = {named_value[0]: named_value[1] for named_value in stats}
        json.dump(stat_dict, out)

@torch.no_grad()
def log_interpolation(G, deformator, step):
    deformator.eval()
    shift_predictor.eval()

    g_samples = validate_g_img(G, deformator)

    grid_g_sample = torchvision.utils.make_grid(g_samples, nrow=int(np.sqrt(len(g_samples))))
    writer.add_image('{}_deformed_interpolation'.format('rand'), grid_g_sample, step)
    torchvision.utils.save_image(g_samples, os.path.join(images_dir, '{}_{}.jpg'.format('rand', step)), 
                                nrow=int(np.sqrt(len(g_samples))), padding=0)

def start_from_checkpoint(deformator, shift_predictor):
    step = 0
    if os.path.isfile(checkpoint):
        state_dict = torch.load(checkpoint)
        step = state_dict['step']
        deformator.load_state_dict(state_dict['deformator'])
        shift_predictor.load_state_dict(state_dict['shift_predictor'])
        print('starting from step {}'.format(step))
    return step

def save_checkpoint(deformator, shift_predictor, step):
    state_dict = {
        'step': step,
        'deformator': deformator.state_dict(),
        'shift_predictor': shift_predictor.state_dict(),
    }
    torch.save(state_dict, checkpoint)

def save_models(deformator, shift_predictor, step):
    torch.save(deformator.state_dict(),
                os.path.join(models_dir, 'deformator_{}.pt'.format(step)))
    torch.save(shift_predictor.state_dict(),
                os.path.join(models_dir, 'shift_predictor_{}.pt'.format(step)))

@torch.no_grad()
def validate_classifier(G, deformator, shift_predictor):
    deformator.eval()
    shift_predictor.eval()
    n_steps = 100

    percents = torch.empty([n_steps])
    for step in range(n_steps):
        z = torch.randn([batch_size] + [max_latent_dim]).cuda()
        target_indices, shifts, basis_shift = make_shifts(deformator.input_dim)

        imgs = G(z)
        imgs_shifted = G.gen_shifted(z, deformator(basis_shift))

        logits, _ = shift_predictor(imgs, imgs_shifted)
        percents[step] = (torch.argmax(logits, dim=1) == target_indices).to(torch.float32).mean()

    return percents.mean()

def log_accuracy(G, deformator, shift_predictor, step):
    deformator.eval()
    shift_predictor.eval()

    accuracy = validate_classifier(G, deformator, shift_predictor)
    writer.add_scalar('accuracy', accuracy.item(), step)

    deformator.train()
    shift_predictor.train()
    return accuracy

def log(G, deformator, shift_predictor, step, avgs):

    deformator.eval()
    shift_predictor.eval()

    if step % steps_per_log == 0:
        log_train(step, True, [avg.flush() for avg in avgs])

    if step % steps_per_img_log == 0:
        log_interpolation(G, deformator, step)

    if step % steps_per_backup == 0 and step > 0:
        save_checkpoint(deformator, shift_predictor, step)
        accuracy = log_accuracy(G, deformator, shift_predictor, step)
        print('Step {} accuracy: {:.3}'.format(step, accuracy.item()))

    if step % steps_per_save == 0 and step > 0:
        save_models(deformator, shift_predictor, step)

    deformator.train()
    shift_predictor.train()

# %%
def train(G, deformator, shift_predictor, multi_gpu=False):
    G.cuda().eval()
    deformator.cuda().train()
    shift_predictor.cuda().train()

    if multi_gpu:
        G = nn.DataParallel(G)
        deformator = nn.DataParallel(deformator)
        shift_predictor = nn.DataParallel(shift_predictor)

    deformator_opt = torch.optim.Adam(deformator.parameters(), lr=deformator_lr) \
        if deformator.type not in [DeformatorType.ID, DeformatorType.RANDOM] else None
    shift_predictor_opt = torch.optim.Adam(
        shift_predictor.parameters(), lr=shift_predictor_lr)

    avgs = MeanTracker('percent'), MeanTracker('loss'), MeanTracker('direction_loss'),\
            MeanTracker('shift_loss')
    avg_correct_percent, avg_loss, avg_label_loss, avg_shift_loss = avgs

    recovered_step = start_from_checkpoint(deformator, shift_predictor)
    for step in range(recovered_step, n_steps, 1):
        G.zero_grad()
        deformator.zero_grad()
        shift_predictor.zero_grad()

        z =  torch.randn([batch_size] + [max_latent_dim]).cuda()
        target_indices, shifts, basis_shift = make_shifts(deformator.input_dim)

        # Deformation
        shift = deformator(basis_shift)

        imgs = G(z)
        imgs_shifted = G.gen_shifted(z, shift)


        logits, shift_prediction = shift_predictor(imgs, imgs_shifted)
        logit_loss = label_weight * CE_loss(logits, target_indices)
        shift_loss = shift_weight * torch.mean(torch.abs(shift_prediction - shifts))

        # total loss
        loss = logit_loss + shift_loss
        loss.backward()

        if deformator_opt is not None:
            deformator_opt.step()
        shift_predictor_opt.step()

        # update statistics trackers
        avg_correct_percent.add(torch.mean(
                (torch.argmax(logits, dim=1) == target_indices).to(torch.float32)).detach())
        avg_loss.add(loss.item())
        avg_label_loss.add(logit_loss.item())
        avg_shift_loss.add(shift_loss)

        log(G, deformator, shift_predictor, step, avgs)

# %%
train(G, deformator, shift_predictor, multi_gpu=False)

# %%



