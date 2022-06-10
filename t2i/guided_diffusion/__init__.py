import lpips, more_itertools, os, tqdm
from einops import rearrange
import numpy as np, torch as th
from PIL import Image
from torchvision.utils import save_image
from neurpy.model.pretrained import load
from neurpy.noise.perlin import *
from neurpy.loss import *
from t2i import T2I
from t2i.crop import *


class GuidedDiffusion(T2I):
    def __init__(self, **kwargs):
        super().__init__()
        self.model, self.diffusion = None, None
        self.P = dict()

    def _generator(self, generator, device='cuda:0', *args, **kwargs):
        self.model, self.model_config, self.diffusion = load(generator, device=device, **kwargs)
        self.side_x = self.side_y = self.model_config['image_size']
        self.lpips_model = lpips_model = lpips.LPIPS(net=kwargs['lpips_model']).to(device)
        if self.model_config['timestep_respacing'].startswith('ddim'):
            self.sample_fn = self.diffusion.ddim_sample_loop_progressive
        else:
            self.sample_fn = self.diffusion.p_sample_loop_progressive
        self.input_shape =  (kwargs['batch_size'], 3, self.side_y, self.side_x)


    # todo: parameterize cutn and keep in t2i/__init__.py
    def _perceptor(self, device='cuda', **kwargs):
        if len(kwargs['p_weights']) != len(kwargs['perceptor']):
            kwargs['p_weights'] = [1] * len(kwargs['perceptor'])

        for perceptor,weight in zip(kwargs['perceptor'], kwargs['p_weights']):
            if not perceptor in self.P.keys():
                model, normalize, tokenize = load(perceptor)
                patch_size = 2 * (model.visual.input_resolution,)
                mean= [0.48145466, 0.45782750, 0.40821073]
                std = [0.26862954, 0.26130258, 0.27577711]
                self.P[perceptor] = dict(perceptor=model, patch_size=patch_size,
                weight=weight, normalize=T.Normalize(mean=mean, std=std),
                tokenize=tokenize, target_embeds=[], weights=[],
                make_cutouts=MakeCutouts_v4(patch_size, kwargs['cutn'], aug=kwargs['aug']))


    def encode_prompts(self, prompt_config, fuzz_steps=25, device='cuda:0', **kwargs):
        for clip_model,clip_params in self.P.items():
            perceptor = clip_params['perceptor']
            for prompt in prompt_config['prompt'].split('|'):

                if '.png' in prompt or '.jpg' in prompt or '.jpeg' in prompt:
                    vals = prompt.rsplit(':', 2)
                    vals = vals + ['', '1', '-inf'][len(vals):]
                    path, weight, stop = vals[0], float(vals[1]), float(vals[2])

                    img = Image.open(path).convert('RGB')
                    rsz = TF.resize(img, min(self.side_x, self.side_y, *img.size), T.InterpolationMode.LANCZOS)
                    ten = TF.to_tensor(rsz).to(device).unsqueeze(0).mul(2).sub(1)
                    batch = clip_params['make_cutouts'](ten)
                    norms = clip_params['normalize'](batch)
                    embed = perceptor.encode_image(norms).float()
                    if kwargs['fuzzy_prompt']:
                        for i in range(fuzz_steps):
                            noise = (embed + th.randn(embed.shape).to(device) * kwargs['rand_mag']).clamp(0,1)
                            clip_params['target_embeds'].append(noise)
                            weights.extend([weight / kwargs['cutn']] * kwargs['cutn'])
                    self.P[clip_model]['target_embeds'].append(embed)
                    self.P[clip_model]['weights'].extend([weight / kwargs['cutn']] * kwargs['cutn'])
                else:
                    vals = prompt.rsplit(':', 2)
                    vals = vals + ['', '1', '-inf'][len(vals):]
                    text, weight, stop = vals[0], float(vals[1]), float(vals[2])
                    txt = perceptor.encode_text(clip_params['tokenize'](text).to(device)).float()

                    if kwargs['fuzzy_prompt']:
                        for i in range(fuzz_steps):
                            noise= (txt + th.randn(txt.shape).cuda() * kwargs['rand_mag']).clamp(0,1)
                            self.P[clip_model]['target_embeds'].append(noise)
                            self.P[clip_model]['weights'].append(weight)
                    else:
                        self.P[clip_model]['target_embeds'].append(txt)
                        self.P[clip_model]['weights'].append(weight)
            #-------------------------------------------------------------------
            self.P[clip_model]['target_embeds'] = th.cat(self.P[clip_model]['target_embeds'])
            self.P[clip_model]['weights'] = th.tensor(self.P[clip_model]['weights'], device=device)
            if self.P[clip_model]['weights'].sum().abs() < 1e-3:
                raise RuntimeError('The weights must not sum to 0.')
            self.P[clip_model]['weights'] /= self.P[clip_model]['weights'].sum().abs()


    def init(self, init=None, perlin=True, perlin_mode=None, device='cuda:0', **kwargs):
        if init and os.path.isfile(init):
            init = Image.open(init).convert('RGB')
            init = init.resize((self.side_x, self.side_y), Image.LANCZOS)
            init = TF.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)
        else:
            size = (self.side_x, self.side_y)
            if perlin_mode == 'color':
                init = create_perlin_noise(size,[1.5**-i*0.5 for i in range(12)], 1, 1, False)
                init2= create_perlin_noise(size,[1.5**-i*0.5 for i in range(8)],  4, 4, False)
            elif perlin_mode == 'gray':
               init = create_perlin_noise(size, [1.5**-i*0.5 for i in range(12)], 1, 1, True)
               init2= create_perlin_noise(size, [1.5**-i*0.5 for i in range(8)],  4, 4, True)
            else:
               init = create_perlin_noise(size, [1.5**-i*0.5 for i in range(12)], 1, 1, False)
               init2= create_perlin_noise(size, [1.5**-i*0.5 for i in range(8)],  4, 4, True)

            init = TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1)
        return init


    def run(self, prompt_config, save_every=1, save_progress=False, device='cuda:0', **kwargs):
        loss_values, target_embeds, weights, output_files = [], [], [], []
        #----------------------------------------------------------------------
        init = self.init(**kwargs)
        init_image = None
        if kwargs['init'] is not None and os.path.isfile(kwargs['init']):
            init_image = kwargs['init']
        #-----------------------------------------------------------------------
        self.encode_prompts(prompt_config, **kwargs)
        #-----------------------------------------------------------------------
        cur_t = None
        def cond_fn(x, t, y=None):
            with th.enable_grad():
                x = x.detach().requires_grad_()
                n = x.shape[0]
                my_t = th.ones([n], device=device, dtype=th.long) * cur_t
                out = self.diffusion.p_mean_variance(self.model, x, my_t, clip_denoised=False, model_kwargs=dict(y=y))
                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out['pred_xstart'] * fac + x * (1 - fac)
                x_in_grad = th.zeros_like(x_in)

                for clip_model,clip_params in self.P.items():
                    for i in range(kwargs['cutn_batches']):
                        clip_in = clip_params['normalize'](clip_params['make_cutouts'](x_in.add(1).div(2)))
                        image_embeds = clip_params['perceptor'].encode_image(clip_in).float()
                        dists = spherical_distance(image_embeds.unsqueeze(1), clip_params['target_embeds'].unsqueeze(0))
                        dists = dists.view([kwargs['cutn'], n, -1])
                        losses = dists.mul(clip_params['weights']).sum(2).mean(0)
                        loss_values.append(losses.sum().item())
                        loss = losses.sum() * kwargs['guidance_loss']
                        x_in_grad += th.autograd.grad(loss, x_in)[0] / kwargs['cutn_batches']

                tv_losses = total_variation_loss(x_in)
                range_losses = range_loss(out['pred_xstart'])
                sat_losses = th.abs(x_in - x_in.clamp(min=-1,max=1)).mean()

                loss = tv_losses.sum() * kwargs['tv_loss'] + \
                range_losses.sum() * kwargs['range_loss'] + \
                sat_losses.sum() * kwargs['sat_loss']

                if init is not None and kwargs['init_weight']:
                    init_losses = self.lpips_model(x_in, init)
                    loss = loss + init_losses.sum() * kwargs['init_weight']

                x_in_grad += th.autograd.grad(loss, x_in)[0]
                grad = -th.autograd.grad(x_in, x, x_in_grad)[0]

            if kwargs['clamp_grad']:
                magnitude = grad.square().mean().sqrt()
                return grad * magnitude.clamp(max=0.05) / magnitude

            return grad
        #-----------------------------------------------------------------------
        sample_args = dict(model_kwargs={}, progress=True, cond_fn=cond_fn,
        init_image=init, skip_timesteps=kwargs['skip_timesteps'],
        randomize_class=kwargs['randomize_class'], clip_denoised=kwargs['clip_denoised'])

        if self.model_config['timestep_respacing'].startswith('ddim'):
            sample_args = dict(**sample_args, eta=eta)
        samples = self.sample_fn(self.model, self.input_shape, **sample_args)
        #-----------------------------------------------------------------------
        cur_t = self.diffusion.num_timesteps - kwargs['skip_timesteps'] - 1
        bar = tqdm.tqdm(total=cur_t)
        for i, sample in enumerate(samples):
            cur_t -= 1
            for j, image in enumerate(sample['pred_xstart']):
                image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))

                if not save_progress: filename = prompt_config['image']
                else:
                    save_n = str((i + 1) * (j + 1)).zfill(6)
                    filename = prompt_config['step'].format(save_n)

                if ((i + 1) * (j + 1) % save_every) == 0 or cur_t == -1:
                    output_file = os.path.join(kwargs['folder'], filename)
                    image.save(output_file)
                    output_files.append(output_file)
        #-----------------------------------------------------------------------
        for p in self.P.keys(): self.P[p]['weights'], self.P[p]['target_embeds'] = [], []
        return output_files
