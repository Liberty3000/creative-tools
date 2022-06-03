import random, torch as th, torchvision as tv
import torchvision.transforms as T
import torch.nn.functional as F

import kornia, kornia.augmentation as K
from kornia.augmentation.base import IntensityAugmentationBase2D

from neurpy.grad import clamp_with_grad
from t2i.util import resample


class FixPadding(th.nn.Module):
    def __init__(self, module=None, threshold=1e-12, noise_frac=0.0):
        super().__init__()
        self.threshold = threshold
        self.noise_frac = noise_frac
        self.module = module

    def forward(self,input):

        dims = input.shape

        if self.module is not None: input = self.module(input + self.threshold)

        light = input.new_empty(dims[0],1,1,1).uniform_(0.,2.)
        mixed = input.view(*dims[:2],-1).sum(dim=1,keepdim=True)

        black = mixed < self.threshold
        black = black.view(-1,1,*dims[2:4]).type(th.float)
        black = kornia.filters.box_blur( black, (5,5) ).clip(0,0.1)/0.1

        mean = input.view(*dims[:2],-1).sum(dim=2) / mixed.count_nonzero(dim=2)
        mean = ( mean[:,:,None,None] * light ).clip(0,1)

        fill = mean.expand(*dims)
        if 0 < self.noise_frac:
            rng = th.get_rng_state()
            fill = fill + th.randn_like(mean) * self.noise_frac
            th.set_rng_state(rng)

        if self.module is not None: input = input - self.threshold

        return th.lerp(input,fill,black)


class RandomNoise(IntensityAugmentationBase2D):
    def __init__(
        self,
        noise: float = 0.1,
        return_transform: bool = False,
        same_on_batch: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch, p_batch=1.0)
        self.frac = noise

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"({super().__repr__()})"

    def generate_parameters(self, shape: th.Size):# -> Dict[str, th.Tensor]:
        noise = th.FloatTensor(1).uniform_(0,self.frac)

        # generate pixel data without throwing off determinism of augs
        rng = th.get_rng_state()
        noise = noise * th.randn(shape)
        th.set_rng_state(rng)

        return dict(noise=noise)

    def apply_transform(self, input, params, transform):
        return input + params['noise'].to(input.device)

#---------------------------------------------------------------------------------------------------------
class MakeCutouts_v1(th.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1., augment=False, aug_noise=0, *args, **kwargs):
        super().__init__()
        self.cut_size = cut_size[0]
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.augment = augment

        self.augs = th.nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomSharpness(0.3, p=0.1),
        K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
        K.RandomPerspective(0.2,p=0.4),
        K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
        K.RandomGrayscale(p=0.15)
        )

        self.noise_fac = aug_noise

    def forward(self, input, *args, **kwargs):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(th.rand([])**self.cut_pow * (max_size - min_size) + min_size)

            offsetx = th.randint(0, sideX - size + 1, ())
            offsety = th.randint(0, sideY - size + 1, ())

            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))

        batch = th.cat(cutouts, dim=0)

        if self.augment: batch = self.augs(batch)

        if self.noise_fac:
            noise = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + noise * th.randn_like(batch)

        return batch

class MakeCutouts_v2(th.nn.Module):
    def __init__(self, cut_size:tuple, cutn, cut_pow=1., full_cuts=False, aug_noise=0, *args, **kwargs):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.full_cuts = full_cuts

        self.augs = th.nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomSharpness(0.3, p=0.1),
        FixPadding(th.nn.Sequential(
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2,p=0.4),
        )),
        K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
        K.RandomGrayscale(p=0.15),
        RandomNoise(noise=aug_noise),
        )

    def forward(self, input, augment=True):
        cut_dim = self.cut_size[0]
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, cut_dim)
        min_size_width = min(sideX, sideY)
        lower_bound = float(cut_dim / min_size_width)

        cutouts = []
        if self.cutn == 1:
            cutouts.append(th.nn.functional.adaptive_avg_pool2d(input, cut_dim))
        elif self.full_cuts:
            cutouts = [F.interpolate(cut, size=self.cut_size) for cut in cutouts]
            cutouts = th.cat(cutouts, dim=0)
        else:
            for _ in range(self.cutn):
                rand = th.zeros(1,).normal_(mean=0.8, std=0.3).clip(lower_bound, 1.)
                size = int(min_size * rand)

                offsetx = th.randint(0, sideX - size + 1, ())
                offsety = th.randint(0, sideY - size + 1, ())

                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                cutouts.append(resample(cutout, self.cut_size))

        batch = th.cat(cutouts, dim=0)
        return batch if not augment else self.augs(batch)


class MakeCutouts_v3(th.nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1., cutn_whole_portion = 0.0, cutn_bw_portion = 0.2, aug_noise=0.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.cutn_whole_portion = cutn_whole_portion
        self.cutn_bw_portion = cutn_bw_portion

        self.augs = th.nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomSharpness(0.3, p=0.1),
        FixPadding(th.nn.Sequential(
            K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),
            K.RandomPerspective(0.2,p=0.4),
        )),
        K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
        K.RandomGrayscale(p=0.15),
        RandomNoise(noise=aug_noise),
        )

    def forward(self, input, augment=True):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size[0])
        cutouts = []
        if self.cutn==1:
            cutouts.append(th.nn.functional.adaptive_avg_pool2d(input, self.cut_size))
            return th.cat(cutouts)

        cut_1 = round(self.cutn*(1-self.cutn_whole_portion))
        cut_2 = self.cutn-cut_1

        gray = tv.transforms.Grayscale(3)

        if cut_1 >0:
            for i in range(cut_1):
                size = int(th.rand([])**self.cut_pow * (max_size - min_size) + min_size)
                offsetx = th.randint(0, sideX - size + 1, ())
                offsety = th.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i < int(self.cutn_bw_portion * cut_1):
                    cutout = gray(cutout)
                cutouts.append(th.nn.functional.adaptive_avg_pool2d(cutout, self.cut_size))
        if cut_2 >0:
            for i in range(cut_2):
                cutout = tv.transforms.functional.rotate(input, angle=random.uniform(-10.0, 10.0), expand=True, fill=rotation_fill)
                if i < int(self.cutn_bw_portion * cut_2):
                    cutout =gray(cutout)
                cutouts.append(th.nn.functional.adaptive_avg_pool2d(cutout, self.cut_size))
        return th.cat(cutouts)

#-------------------------------------------------------------------------------
class MakeCutouts_Dango(th.nn.Module):
    '''


    '''
    def __init__(self, cut_size, Overview=4, WholeCrop = 0, WC_Allowance = 10,
    WC_Grey_P=0.2, InnerCrop = 0, IC_Size_Pow=0.5, IC_Grey_P = 0.2):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.WholeCrop= WholeCrop
        self.WC_Allowance = WC_Allowance
        self.WC_Grey_P = WC_Grey_P
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P

        self.augs = [
        T.RandomHorizontalFlip(p=0.5),
        T.Lambda(lambda x: x + th.randn_like(x) * 0.01),
        T.RandomAffine(degrees=5, translate=(0.05, 0.05),
        fill=-1,  interpolation = T.InterpolationMode.BILINEAR, ),
        T.Lambda(lambda x: x + th.randn_like(x) * 0.01),
        T.Lambda(lambda x: x + th.randn_like(x) * 0.01),
        T.RandomGrayscale(p=0.1),
        T.Lambda(lambda x: x + th.randn_like(x) * 0.01),
        T.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05),
        ]
        self.transform = T.Compose(self.augs)
        self.pad_args = dict(mode='constant', value=-1)

    def forward(self, input):
        from resize_right import resize, calc_pad_as

        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        l_size = max(sideX, sideY)
        output_shape = [1,3,self.cut_size,self.cut_size]
        output_shape_2 = [1,3,self.cut_size+2,self.cut_size+2]
        x = (sideY-max_size)//2+round(max_size*0.05)
        y = (sideY-max_size)//2+round(max_size*0.05)
        z = (sideX-max_size)//2+round(max_size*0.05)
        w = (sideX-max_size)//2+round(max_size*0.05)
        pad_input = F.pad(input,(w,x,y,z), **self.pad_args)
        cutouts_list = []

        if self.Overview > 0:
            cutouts = []
            cutout = resize(pad_input, out_shape=output_shape)
            if self.Overview in [1,2,4]:
                if self.Overview >= 2:
                    cutout = th.cat((cutout,gray(cutout)))
                if self.Overview==4:
                    cutout = th.cat((cutout, TF.hflip(cutout)))
            else:
                output_shape_all = list(output_shape)
                output_shape_all[0]=self.Overview
                cutout = resize(pad_input, out_shape=output_shape_all)
                if aug: cutout=self.augs(cutout)
            cutouts_list.append(cutout)

        if self.InnerCrop >0:
            cutouts=[]
            for i in range(self.InnerCrop):
                size = int(th.rand([])**self.IC_Size_Pow * (max_size - min_size) + min_size)
                offsetx = th.randint(0, sideX - size + 1, ())
                offsety = th.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            cutouts_tensor = th.cat(cutouts)
            cutouts = []
            cutouts_list.append(cutouts_tensor)
        cutouts = th.cat(cutouts_list)
        return cutouts
