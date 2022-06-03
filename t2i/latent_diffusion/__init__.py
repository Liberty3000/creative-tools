from einops import rearrange
import more_itertools, tqdm
from PIL import Image
import numpy as np, torch as th
from torchvision.utils import save_image
from neurpy.model.pretrained import load
from t2i import T2I


class LatentDiffusion(T2I):
    def __init__(self, **kwargs):
        super().__init__()
        self.model, self.sampler = None, None

    def _generator(self, generator, *args, **kwargs):
        self.model, self.sampler = load(generator, **kwargs)

    def run(self, prompt_config, save_every=1, save_progress=False, device='cuda:0', **kwargs):

        prompt = prompt_config['prompt']
        output_files = []

        with th.no_grad():
            with self.model.ema_scope():

                n_chunk = kwargs['seeds'] // kwargs['batch_size']
                chunked = more_itertools.chunked(range(kwargs['seeds']), kwargs['batch_size'])
                chunks  = tqdm.tqdm(enumerate(chunked), total=n_chunk, desc = '      epochs', position=0, leave=True)
                for i,chunk in chunks:
                    batch_size = len(chunk)

                    uc = None
                    if kwargs['scale'] != 1.0:
                        uc = self.model.get_learned_conditioning(batch_size * [''])

                    for n in tqdm.trange(kwargs['n_iter']):
                        input = batch_size * [prompt]
                        c = self.model.get_learned_conditioning(input)
                        shape = [4, kwargs['image_h'] // 8, kwargs['image_w'] // 8]
                        samples_ddim, _ = self.sampler.sample(conditioning=c, shape=shape,
                                                         S=kwargs['ddim_steps'],
                                                         batch_size=batch_size,
                                                         unconditional_conditioning=uc,
                                                         unconditional_guidance_scale=kwargs['scale'],
                                                         eta=kwargs['ddim_eta'], verbose=False)

                        outputs = self.model.decode_first_stage(samples_ddim)
                        outputs = th.clamp((outputs + 1.0) / 2.0, min=0.0, max=1.0)

                        for output in outputs:
                            output = 255. * rearrange(output.cpu().numpy(), 'c h w -> h w c')
                            image = Image.fromarray(output.astype(np.uint8))

                            output_file = f'{prompt_config["seeded"]}.{len(output_files) + 1}.png'
                            image.save(output_file)
                            output_files.append(output_file)

        return output_files
