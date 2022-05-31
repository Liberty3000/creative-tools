import os, tqdm
import numpy as np
from PIL import Image
import torch as th, torch.nn.functional as F
import torchvision as tv, torchvision.transforms as T
from torchvision.utils import save_image
from torchmetrics import SSIM

from neurpy.grad import clamp_with_grad, replace_grad
from neurpy.loss import total_variation_loss
from neurpy.model.pretrained import load
from neurpy.noise import random_perlin, random_pyramid
from neurpy.util import enforce_reproducibility

from t2i import T2I
from t2i.crop import *
from t2i.vqlipse.util import refinement


def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)


class Prompt(th.nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', th.as_tensor(weight))
        self.register_buffer('stop', th.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)

        dists = input_normed.sub(embed_normed).norm(dim=2)

        dists = dists.div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        grads = replace_grad(dists, th.maximum(dists, self.stop)).mean()

        return self.weight.abs() * grads


class VQLIPSE(T2I):

    def __init__(self, **kwargs):
        super().__init__()
        self.G, self.P = None, dict()

    def _generator(self, generator, image_w, image_h, device='cuda', *args, **kwargs):

        from taming.models.vqgan import GumbelVQ

        self.G = load(generator, image_size=(image_w, image_h))[0]
        self.G.eval().requires_grad_(False)

        if isinstance(self.G, GumbelVQ):
            self.z_min = self.G.quantize.emb.weight.min(dim=0).values[None,:,None,None]
            self.z_max = self.G.quantize.emb.weight.max(dim=0).values[None,:,None,None]
            self.e_dim, self.n_tok = 256, self.G.quantize.n_embed
        else:
            self.z_min = self.G.quantize.embedding.weight.min(dim=0).values[None,:,None,None]
            self.z_max = self.G.quantize.embedding.weight.max(dim=0).values[None,:,None,None]
            self.e_dim, self.n_tok = self.G.quantize.e_dim, self.G.quantize.n_e

        self.f = 2**(self.G.decoder.num_resolutions - 1)
        self.G = self.G.to(device)


    def generate(self, z, device='cuda'):

        from taming.models.vqgan import GumbelVQ

        if isinstance(self.G, GumbelVQ): weight = self.G.quantize.embed.weight
        else: weight = self.G.quantize.embedding.weight
        z = vector_quantize(z.movedim(1, 3), weight).movedim(3, 1)
        output = self.G.decode(z).add(1).div(2)
        return output


    def init_z(self, init=None, image_w=None, image_h=None, device='cuda', verbose=False, **kwargs):

        self.toks_x, self.toks_y = image_w // self.f, image_h // self.f
        self.side_x, self.side_y = self.toks_x * self.f, self.toks_y * self.f
        ydim, xdim = self.toks_y * self.f, self.toks_x * self.f

        if init is not None and os.path.isfile(init):
            pil_image = Image.open(init).convert('RGB')

            pil_image = pil_image.resize((self.side_x, self.side_y), Image.LANCZOS)
            input_ = tv.transforms.functional.to_tensor(pil_image).to(device)
            input_ = input_.unsqueeze(0) * 2 - 1
            z, *_ = self.G.encode(input_)

            if verbose: print(f'init :: {z.shape} :: image :: {image}')
        elif init == 'perlin':
            perlin_weight, perlin_octaves = kwargs['weight'], kwargs['octaves']
            rand_init = random_perlin((ydim, xdim), perlin_weight, perlin_octaves)

            z, *_ = self.G.encode(rand_init.to(device) * 2 - 1)
            if verbose: print(f'init :: {z.shape} :: perlin')
        elif init == 'pyramid':
            pyramid_decay, pyramid_octaves = kwargs['decay'], kwargs['octaves']
            rand_init = random_pyramid((1, 3, ydim, xdim), pyramid_octaves, pyramid_decay)
            rand_init = (rand_init * 0.5 + 0.5).clip(0, 1)

            z, *_ = self.G.encode(rand_init.to(device) * 2 - 1)
            if verbose: print(f'init :: {z.shape} :: pyramid')
        else:
            from taming.models.vqgan import GumbelVQ
            rand_int = th.randint(self.n_tok, [self.toks_y * self.toks_x], device=device)
            one_hot = F.one_hot(rand_int, self.n_tok).float()
            if isinstance(self.G, GumbelVQ): emb = self.G.quantize.embed.weight
            else: emb = self.G.quantize.embedding.weight
            z = one_hot @ emb
            z = z.view([-1, self.toks_y, self.toks_x, self.e_dim]).permute(0, 3, 1, 2)
            if verbose: print(f'init :: {z.shape} :: random')

        return z


    def encode_text(self, raw, p_weights=None, device='cuda:0', verbose=False):
        prompts = raw.split('|')
        for i,prompt in enumerate(prompts):
            vals = prompt.rsplit(':', 2)
            vals = vals + ['', '1', '-inf'][len(vals):]
            text, weight, stop = vals[0], float(vals[1]), float(vals[2])
            for p,p_dict in self.P.items():
                perceptor, patch_size = p_dict['perceptor'], p_dict['patch_size']
                normalize, tokenize = p_dict['normalize'], p_dict['tokenize']
                weight *= p_dict['weight']

                if os.path.exists(text):
                    image_tensor = Image.open(text).resize(2 * (patch_size,))
                    img = T.ToTensor()(image_tensor.convert('RGB')).unsqueeze(0)
                    emb = perceptor.encode_image(normalize(img).to(device)).float()
                else:
                    tok = tokenize(text).to(device)
                    emb = perceptor.encode_text(tok).float()

                prompt_ = Prompt(emb, weight=weight, stop=stop)
                self.P[p]['prompts'].append(prompt_)


    def forward(self, **kwargs):
        output, losses = self.generate(self.z), []
        #-----------------------------------------------------------------------
        # todo: load `cutm` from module, vary cutouts per-perceptor, etc.
        patch_size = 2 * (list(self.P.values())[0]['patch_size'],)
        if kwargs['cutm'] == 'v2':
            self.make_cutouts = MakeCutouts_v2(patch_size, cutn=kwargs['cutn'],
            cutp=kwargs['cutp'], aug_noise=kwargs['aug_noise'])
        if kwargs['cutm'] == 'v3':
            self.make_cutouts = MakeCutouts_v3(cut_size=patch_size, cutn=kwargs['cutn'],
            cut_pow=kwargs['cutp'], aug_noise=kwargs['aug_noise'])

        cutouts = self.make_cutouts(output, augment=kwargs['aug'])
        cutouts = clamp_with_grad(cutouts, 0, 1)
        #-----------------------------------------------------------------------
        p_losses = dict()
        for k,v in self.P.items():
            if v['patch_size'] != cutouts.size(-1):
                cutouts = F.interpolate(cutouts, 2 * (v['patch_size'],))

            perceptor, inputs = v['perceptor'], v['normalize'](cutouts)
            encoding = perceptor.encode_image(inputs).float()
            for i,prompt in enumerate(v['prompts']):
                losses.append(prompt(encoding))
                p_losses[f'{k}-prompts[{i}]'] = losses[-1]
        #-----------------------------------------------------------------------
        tv_loss, ssim_loss, init_loss = 0.,0.,0.
        if 'tv_loss' in kwargs.keys() and kwargs['tv_loss']:
            tv_loss = kwargs['tv_loss'] * total_variation_loss(output)
            losses.append(tv_loss)

        if 'ssim_loss' in kwargs.keys() and kwargs['ssim_loss']:
            targ = self.generate(self.init_z(kwargs['image']))
            pred = self.generate(self.get_z())
            ssim_loss = SSIM()(pred, targ)
            losses.append(kwargs['ssim_loss'] * ssim_loss)

        if 'init_weight' in kwargs.keys() and kwargs['init_weight']:
            init_loss = F.mse_loss(self.z, self.z_init) * kwargs['init_weight'] / 2
            losses.append(init_loss)
        #-----------------------------------------------------------------------
        return dict(image=output, losses=losses, prompt_losses=p_losses)


    def training_step(self, cutn_batches=1, gradient_accumulate=1, *args, **kwargs):
        total_loss = 0.
        #-----------------------------------------------------------------------
        for cutn_batch in range(cutn_batches):
            for _ in range(gradient_accumulate):
                outputs = self.forward(*args, **kwargs)
                loss = sum(outputs['losses']) / gradient_accumulate
                loss.backward()

            with th.no_grad():
                self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

            self.optim.step()
            self.optim.zero_grad(set_to_none=True)
        #-----------------------------------------------------------------------
        image = outputs['image'].detach().cpu()
        output_dict = dict(image=image, loss=loss.item())
        #-----------------------------------------------------------------------
        if kwargs['zoom_2d'] and kwargs['zoom_step'] and kwargs['batch_nb'] > \
           kwargs['zoom_init'] and kwargs['batch_nb'] % kwargs['zoom_step'] == 0:
            import scipy.ndimage as nd

            zoom_transform = ([1 - kwargs['zoom_rate'], 1 - kwargs['zoom_rate'], 1],
                              [kwargs['image_h'] * kwargs['zoom_rate'] / 2,
                               kwargs['image_w'] * kwargs['zoom_rate'] / 2, 0])

            image = np.asarray(T.ToPILImage()(image.squeeze()).convert('RGB'))
            zoomed = nd.affine_transform(image, *zoom_transform, order=1)
            output = T.ToPILImage()(zoomed)
            output.save('zoom.png')
            self.optimizer(self.init_z(init='zoom.png', **kwargs), **kwargs)
            output_dict['animated'] = T.ToTensor()(zoomed)
        elif kwargs['zoom_3d'] and kwargs['zoom_step'] and kwargs['batch_nb'] > \
           kwargs['zoom_init'] and kwargs['batch_nb'] % kwargs['zoom_step'] == 0:
            from t2i.animate import zoom_3d, parametric_eval
            translation = (kwargs['translate_x'], kwargs['translate_y'], kwargs['translate_z'])
            t = kwargs['zoom_init']-kwargs['batch_nb']/(kwargs['zoom_step']*kwargs['fps'])
            input = T.ToPILImage()(image.squeeze())
            pil_image = zoom_3d(input, translate=translation, t=t,
                                rotate=kwargs['rotate_3d'], **kwargs)
            pil_image.save('zoom.png')
            self.optimizer(self.init_z(init='zoom.png', **kwargs), **kwargs)
            output_dict['animated'] = T.ToTensor()(pil_image)

        #-----------------------------------------------------------------------
        return output_dict
