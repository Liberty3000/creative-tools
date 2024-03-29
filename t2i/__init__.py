import uuid, tqdm
import torch as th
import torchvision.transforms as T
from torchvision.utils import save_image
from neurpy.model.pretrained import load


def config_prompt(prompt, seed=None, step=False, max_chars=255):
    conf = dict(id=str(uuid.uuid4())[:8], prompt=prompt, seed=seed,
    formatter='{prompt}.{seed}.{id}.{step}.png', tokens=[])
    #---------------------------------------------------------------------------
    prompts = prompt.split('||')
    if len(prompts) > 1:
        prompt = '{}...{}'.format(prompts[0][:32], prompts[-1][-32:])
    prompt = prompt [:246]
    #---------------------------------------------------------------------------
    filtered = prompt.rstrip()
    for char in [' ', '-', ',']: filtered = filtered.replace(char, '_')
    for p in filtered.split('_'): conf['tokens'].append(p)
    #---------------------------------------------------------------------------
    filtered = filtered[:max_chars]
    seeded = f'{filtered}.{str(seed).zfill(6)}.{conf["id"]}'
    conf['filtered'], conf['seeded'] = filtered, seeded
    conf['image'] =  f'{conf["seeded"]}.png'
    if step:
        conf['step'] = f'{seeded}' + '.{}.png'
    #---------------------------------------------------------------------------
    files = dict(video='.mp4',
                 depth='-depth.png',
                 swirl='-swirl.png',
                 zoom='-zoom.png',
                 latent='.pt')
    #---------------------------------------------------------------------------
    extension = map(lambda ext:f'{seeded}{ext}', files.values())
    return dict(**conf, **dict(zip(files.keys(), extension)))


class T2I(th.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def init_z(self, **kwargs):
        raise NotImplementedError

    def get_z(self):
        return self.z

    def set_z(self, z):
        self.z = z

    def get_lr(self):
        return [g['lr'] for g in self.optim.param_groups][0]

    def set_lr(self, lr):
        for g in self.optim.params:
            g['lr'] = lr
    #---------------------------------------------------------------------------
    def _generator(self, generator, device='cuda', **kwargs):
        G = load(generator, **kwargs)[0]
        G.eval().requires_grad_(False)
        self.G = G.to(device)
        return dict(G=self.G)

    def _perceptor(self, device='cuda', **kwargs):
        if len(kwargs['p_weights']) != len(kwargs['perceptor']):
            kwargs['p_weights'] = [1] * len(kwargs['perceptor'])

        for perceptor,weight in zip(kwargs['perceptor'], kwargs['p_weights']):
            if not perceptor in self.P.keys():
                model, normalize, tokenize = load(perceptor)
                patch_size = model.visual.input_resolution
                mean= [0.48145466, 0.45782750, 0.40821073]
                std = [0.26862954, 0.26130258, 0.27577711]
                self.P[perceptor] = dict(perceptor=model, patch_size=patch_size,
                weight=weight, normalize=T.Normalize(mean=mean, std=std),
                tokenize=tokenize, prompts=[])
            self.P[perceptor]['weight'] = weight
            self.P[perceptor]['prompts'] = []
    #---------------------------------------------------------------------------

    def optimizer(self, z, lr, weight_decay=0., ema_decay=None, optimizer='Adam', **kwargs):
        self.z = z
        self.z_init = self.z.clone()
        self.z.requires_grad_(True)
        if ema_decay is not None: self.z = EMA(self.z, ema_decay)
        Opt = getattr(th.optim, optimizer)
        self.optim = Opt([self.z], lr=lr, weight_decay=weight_decay)


    def generate(self, z):
        raise NotImplementedError


    def forward(self, *args, **kwargs):
        image = self.generate(self.get_z())
        return dict(image=image, loss=0.)


    def training_step(self, cutn_batches=1, grad_accumulate=1,  *args, **kwargs):
        total_loss=0.
        for cutn_batch in range(cutn_batches):
            self.optim.zero_grad(set_to_none=True)
            for grad_step in range(grad_accumulate):
                outputs = self.forward(*args, **kwargs)
                loss = sum(outputs['losses']) / grad_accumulate
                total_loss += loss.item()
                loss.backward()
            self.optim.step()
        total_loss /= (cutn_batch * gradient_accumulate)
        return dict(**outputs, loss=loss.item(), total_loss=total_loss)


    def run(self, prompt_config, steps=150, save_every=10, save_progress=True,
            preview=False, verbose=False, **kwargs):
        output_files = []
        scheduler = None

        prompt = prompt_config['prompt']
        prompts = prompt.split('||')
        epochs = len(prompts)
        updates = (epochs * steps) // save_every
        banner = 'loss: {:04.2f}, lr: {:.5f}'

        bar = tqdm.tqdm(total=updates, desc='image update', position=2, leave=True)
        for epoch in tqdm.trange(epochs, desc = '      epochs', position=0, leave=True):
            itrs = tqdm.trange(steps, desc='   iteration', position=1, leave=True)
            bar.update(0)

            self.encode_text(prompts[epoch], verbose=verbose)

            for step in itrs:
                batch_nb = ((1 + epoch) * step)
                outputs = self.training_step(prompt=prompts[epoch], batch_nb=batch_nb, **kwargs)

                if batch_nb and batch_nb % save_every == 0:
                    bar.update(1)

                    if save_progress:
                        save_n = str(batch_nb).zfill(6)
                        output_file = prompt_config['step'].format(save_n)
                    else:  output_file = prompt_config['image']
                    output_files.append(output_file)

                    with th.no_grad():
                        if not 'animated' in outputs.keys():
                            save_image(outputs['image'], output_file)
                        else:
                            save_image(outputs['animated'], output_file)

                if scheduler is not None: scheduler.step()

                if not kwargs['aug']: desc = banner
                else: desc = banner + ', aug: {:.2f}'.format(kwargs['aug_noise'])
                itrs.set_description(desc.format(outputs['loss'], self.get_lr()))
                if preview and len(output_files): open_window(output_files[-1])

        print(output_files[-1])
        return output_files
