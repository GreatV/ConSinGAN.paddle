import paddle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
from skimage import io as img
from skimage import color, morphology, filters
import imageio
import os
import random
import datetime
import dateutil.tz
import copy
from albumentations import HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, OneOf, Compose, MultiplicativeNoise, ToSepia, ChannelDropout, ChannelShuffle, Cutout, InvertImg
from ConSinGAN.imresize import imresize, imresize_in, imresize_to_shape


def denorm(x):
    out = (x + 1) / 2
    return out.clip(min=0, max=1)


def norm(x):
    out = (x - 0.5) * 2
    return out.clip(min=-1, max=1)


def convert_image_np(inp):
    if inp.shape[1] == 3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1, :, :, :])
        inp = inp.numpy().transpose((1, 2, 0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1, -1, :, :])
        inp = inp.numpy().transpose((0, 1))
    inp = np.clip(inp, 0, 1)
    return inp


def generate_noise(size, num_samp=1, device='cuda', type='gaussian', scale=1):
    if type == 'gaussian':
        noise = paddle.randn(shape=[num_samp, size[0], round(size[1] /
            scale), round(size[2] / scale)])
        noise = upsampling(noise, size[1], size[2])
    elif type == 'gaussian_mixture':
        noise1 = paddle.randn(shape=[num_samp, size[0], size[1], size[2]]) + 5
        noise2 = paddle.randn(shape=[num_samp, size[0], size[1], size[2]])
        noise = noise1 + noise2
    elif type == 'uniform':
        noise = paddle.randn(shape=[num_samp, size[0], size[1], size[2]])
    else:
        raise NotImplementedError
    return noise


def upsampling(im, sx, sy):
    m = paddle.nn.Upsample(size=[round(sx), round(sy)], mode='bilinear',
        align_corners=True)
    return m(im)


def move_to_gpu(t):
    if paddle.device.cuda.device_count() >= 1:
        t = t.to(str('cuda').replace('cuda', 'gpu'))
    return t


def move_to_cpu(t):
    t = t.to(str('cpu').replace('cuda', 'gpu'))
    return t


def save_image(name, image):
    plt.imsave(name, convert_image_np(image), vmin=0, vmax=1)


def sample_random_noise(depth, reals_shapes, opt):
    noise = []
    for d in range(depth + 1):
        if d == 0:
            noise.append(generate_noise([opt.nc_im, reals_shapes[d][2],
                reals_shapes[d][3]], device=opt.place).detach())
        elif opt.train_mode == 'generation' or opt.train_mode == 'animation':
            noise.append(generate_noise([opt.nfc, reals_shapes[d][2] + opt.
                num_layer * 2, reals_shapes[d][3] + opt.num_layer * 2],
                device=opt.place).detach())
        else:
            noise.append(generate_noise([opt.nfc, reals_shapes[d][2],
                reals_shapes[d][3]], device=opt.place).detach())
    return noise


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    MSGGan = False
    if MSGGan:
        alpha = paddle.rand(shape=[1, 1])
        alpha = alpha.to(device)
        interpolates = [(alpha * rd + (1 - alpha) * fd) for rd, fd in zip(
            real_data, fake_data)]
        interpolates = [i.to(device) for i in interpolates]
        disc_interpolates = netD(interpolates)
    else:
        alpha = paddle.rand(shape=[1, 1])
        alpha = alpha.expand(shape=real_data.shape)
        alpha = alpha.to(device)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.to(device)
        disc_interpolates = netD(interpolates)
        gradients = disc_interpolates.grad()
    gradient_penalty = ((gradients.norm(p=2, axis=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def read_image(opt):
    x = img.imread('%s' % opt.input_name)
    x = np2torch(x, opt)
    x = x[:, 0:3, :, :]
    return x


def read_image_dir(dir, opt):
    x = img.imread(dir)
    x = np2torch(x, opt)
    x = x[:, 0:3, :, :]
    return x


def np2torch(x, opt):
    if opt.nc_im == 3:
        x = x[:, :, :, None]
        x = x.transpose((3, 2, 0, 1)) / 255
    else:
        x = color.rgb2gray(x)
        x = x[:, :, None, None]
        x = x.transpose(3, 2, 0, 1)
    x = paddle.to_tensor(data=x)
    if not opt.not_cuda:
        x = move_to_gpu(x)
    x = x.astype(paddle.Tensor) if not opt.not_cuda else x.astype(paddle.Tensor
        )
    x = norm(x)
    return x


def torch2uint8(x):
    x = x[0, :, :, :]
    x = x.transpose(perm=(1, 2, 0))
    x = 255 * denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x


def read_image2np(opt):
    x = img.imread('%s' % opt.input_name)
    x = x[:, :, 0:3]
    return x


def save_networks(netG, netDs, z, opt):
    paddle.save(obj=netG.state_dict(), path='%s/netG.pth' % opt.outf)
    if isinstance(netDs, list):
        for i, netD in enumerate(netDs):
            paddle.save(obj=netD.state_dict(), path='%s/netD_%s.pth' % (opt
                .outf, str(i)))
    else:
        paddle.save(obj=netDs.state_dict(), path='%s/netD.pth' % opt.outf)
    paddle.save(obj=z, path='%s/z_opt.pth' % opt.outf)


def adjust_scales2image(real_, opt):
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]), 1)
    real = imresize(real_, opt.scale1, opt)
    opt.stop_scale = opt.train_stages - 1
    opt.scale_factor = math.pow(opt.min_size / min(real.shape[2], real.
        shape[3]), 1 / opt.stop_scale)
    return real


def create_reals_pyramid(real, opt):
    reals = []
    if opt.train_mode == 'harmonization':
        for i in range(opt.stop_scale):
            scale = math.pow(opt.scale_factor, opt.stop_scale - i)
            curr_real = imresize(real, scale, opt)
            reals.append(curr_real)
    else:
        for i in range(opt.stop_scale):
            scale = math.pow(opt.scale_factor, (opt.stop_scale - 1) / math.
                log(opt.stop_scale) * math.log(opt.stop_scale - i) + 1)
            curr_real = imresize(real, scale, opt)
            reals.append(curr_real)
    reals.append(real)
    return reals


def load_trained_model(opt):
    dir = generate_dir2save(opt)
    if os.path.exists(dir):
        Gs = paddle.load(path='%s/Gs.pth' % dir)
        Zs = paddle.load(path='%s/Zs.pth' % dir)
        reals = paddle.load(path='%s/reals.pth' % dir)
        NoiseAmp = paddle.load(path='%s/NoiseAmp.pth' % dir)
    else:
        print('no trained model exists: {}'.format(dir))
    return Gs, Zs, reals, NoiseAmp


def generate_dir2save(opt):
    training_image_name = opt.input_name[:-4].split('/')[-1]
    dir2save = 'TrainedModels/{}/'.format(training_image_name)
    dir2save += opt.timestamp
    dir2save += '_{}'.format(opt.train_mode)
    if opt.train_mode == 'harmonization' or opt.train_mode == 'editing':
        if opt.fine_tune:
            dir2save += '_{}'.format('fine-tune')
    dir2save += '_train_depth_{}_lr_scale_{}'.format(opt.train_depth, opt.
        lr_scale)
    if opt.batch_norm:
        dir2save += '_BN'
    dir2save += '_act_' + opt.activation
    if opt.activation == 'lrelu':
        dir2save += '_' + str(opt.lrelu_alpha)
    return dir2save


def post_config(opt):
    opt.place = str('cpu' if opt.not_cuda else 'cuda:{}'.format(opt.gpu)
        ).replace('cuda', 'gpu')
    opt.noise_amp_init = opt.noise_amp
    opt.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime(
        '%Y_%m_%d_%H_%M_%S')
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    paddle.seed(seed=opt.manualSeed)
    if paddle.device.cuda.device_count() >= 1 and opt.not_cuda:
        print(
            'WARNING: You have a CUDA device, so you should probably run with --cuda'
            )
    return opt


def load_config(opt):
    if not os.path.exists(opt.model_dir):
        print('Model not found: {}'.format(opt.model_dir))
        exit()
    with open(os.path.join(opt.model_dir, 'parameters.txt'), 'r') as f:
        params = f.readlines()
        for param in params:
            param = param.split('-')
            param = [p.strip() for p in param]
            param_name = param[0]
            param_value = param[1]
            try:
                param_value = int(param_value)
            except ValueError:
                try:
                    param_value = float(param_value)
                except ValueError:
                    pass
            setattr(opt, param_name, param_value)
    return opt


def dilate_mask(mask, opt):
    if opt.train_mode == 'harmonization':
        element = morphology.disk(radius=7)
    if opt.train_mode == 'editing':
        element = morphology.disk(radius=20)
    mask = torch2uint8(mask)
    mask = mask[:, :, 0]
    mask = morphology.binary_dilation(mask, selem=element)
    mask = filters.gaussian(mask, sigma=5)
    nc_im = opt.nc_im
    opt.nc_im = 1
    mask = np2torch(mask, opt)
    opt.nc_im = nc_im
    mask = mask.expand(shape=[1, 3, mask.shape[2], mask.shape[3]])
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask


def shuffle_grid(image, max_tiles=5):
    tiles = []
    img_w, img_h = image.shape[0], image.shape[1]
    _max_tiles = random.randint(1, max_tiles)
    if _max_tiles == 1:
        w_min, h_min = int(img_w * 0.2), int(img_h * 0.2)
        w_max, h_max = int(img_w * 0.5), int(img_h * 0.5)
        x_translation_min, y_translation_min = int(img_w * 0.05), int(img_h *
            0.05)
        x_translation_max, y_translation_max = int(img_w * 0.15), int(img_h *
            0.15)
    elif _max_tiles == 2:
        w_min, h_min = int(img_w * 0.15), int(img_h * 0.15)
        w_max, h_max = int(img_w * 0.3), int(img_h * 0.3)
        x_translation_min, y_translation_min = int(img_w * 0.05), int(img_h *
            0.05)
        x_translation_max, y_translation_max = int(img_w * 0.1), int(img_h *
            0.1)
    elif _max_tiles == 3:
        w_min, h_min = int(img_w * 0.1), int(img_h * 0.1)
        w_max, h_max = int(img_w * 0.2), int(img_h * 0.2)
        x_translation_min, y_translation_min = int(img_w * 0.05), int(img_h *
            0.05)
        x_translation_max, y_translation_max = int(img_w * 0.1), int(img_h *
            0.1)
    else:
        w_min, h_min = int(img_w * 0.1), int(img_h * 0.1)
        w_max, h_max = int(img_w * 0.15), int(img_h * 0.15)
        x_translation_min, y_translation_min = int(img_w * 0.05), int(img_h *
            0.05)
        x_translation_max, y_translation_max = int(img_w * 0.1), int(img_h *
            0.1)
    for _ in range(_max_tiles):
        x, y = random.randint(0, img_w), random.randint(0, img_h)
        w, h = random.randint(w_min, w_max), random.randint(h_min, h_max)
        if x + w >= img_w:
            w = img_w - x
        if y + h >= img_h:
            h = img_h - y
        x_t, y_t = random.randint(x_translation_min, x_translation_max
            ), random.randint(y_translation_min, y_translation_max)
        if random.random() < 0.5:
            x_t, y_t = -x_t, -y_t
            if x + x_t < 0:
                x_t = -x
            if y + y_t < 0:
                y_t = -y
        else:
            if x + x_t + w >= img_w:
                x_t = img_w - w - x
            if y + y_t + h >= img_h:
                y_t = img_h - h - y
        tiles.append([x, y, w, h, x + x_t, y + y_t])
    new_image = copy.deepcopy(image)
    for tile in tiles:
        x, y, w, h, x_new, y_new = tile
        new_image[x_new:x_new + w, y_new:y_new + h, :] = image[x:x + w, y:y +
            h, :]
    return new_image


class Augment:

    def __init__(self):
        super().__init__()
        self._transofrm = self.strong_aug()

    def strong_aug(self):
        color_r = random.randint(0, 256)
        color_g = random.randint(0, 256)
        color_b = random.randint(0, 256)
        num_holes = random.randint(1, 2)
        if num_holes == 2:
            max_h_size = random.randint(15, 30)
            max_w_size = random.randint(15, 30)
        else:
            max_h_size = random.randint(30, 60)
            max_w_size = random.randint(30, 60)
        return Compose([OneOf([OneOf([MultiplicativeNoise(multiplier=[0.5, 
            1.5], elementwise=True, per_channel=True, p=0.2),
            IAAAdditiveGaussianNoise(), GaussNoise()]), OneOf([InvertImg(),
            ToSepia()]), OneOf([ChannelDropout(channel_drop_range=(1, 1),
            fill_value=0), ChannelShuffle()]), HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=
            0.1)], p=0.25), Cutout(num_holes=num_holes, max_h_size=
            max_h_size, max_w_size=max_w_size, fill_value=[color_r, color_g,
            color_b], p=0.9)])

    def transform(self, **x):
        _transform = self.strong_aug()
        return _transform(**x)


def generate_gif(dir2save, netG, fixed_noise, reals, noise_amp, opt, alpha=
    0.1, beta=0.9, start_scale=1, num_images=100, fps=10):

    def denorm_for_gif(img):
        img = denorm(img).detach()
        img = img[0, :, :, :].cpu().numpy()
        img = img.transpose(1, 2, 0) * 255
        img = img.astype(np.uint8)
        return img
    reals_shapes = [r.shape for r in reals]
    all_images = []
    with paddle.no_grad():
        noise_random = sample_random_noise(len(fixed_noise) - 1,
            reals_shapes, opt)
        z_prev1 = [(0.99 * fixed_noise[i] + 0.01 * noise_random[i]) for i in
            range(len(fixed_noise))]
        z_prev2 = fixed_noise
        for _ in range(num_images):
            noise_random = sample_random_noise(len(fixed_noise) - 1,
                reals_shapes, opt)
            diff_curr = [(beta * (z_prev1[i] - z_prev2[i]) + (1 - beta) *
                noise_random[i]) for i in range(len(fixed_noise))]
            z_curr = [(alpha * fixed_noise[i] + (1 - alpha) * (z_prev1[i] +
                diff_curr[i])) for i in range(len(fixed_noise))]
            if start_scale > 0:
                z_curr = [fixed_noise[i] for i in range(start_scale)] + [z_curr
                    [i] for i in range(start_scale, len(fixed_noise))]
            z_prev2 = z_prev1
            z_prev1 = z_curr
            sample = netG(z_curr, reals_shapes, noise_amp)
            sample = denorm_for_gif(sample)
            all_images.append(sample)
    imageio.mimsave('{}/start_scale={}_alpha={}_beta={}.gif'.format(
        dir2save, start_scale, alpha, beta), all_images, fps=fps)
