import click, gc, os, random, pprint, yaml
import neptune.new as neptune
import torch as th
from t2i import config_prompt
from t2i.bigsleep import BigSleep as T2I
from t2i.util import enforce_reproducibility


@click.option(              '--seed', default=random.randint(0,1e6))
@click.option(             '--seeds', default=None, type=int)
@click.option(        '--experiment', default='liberty3000/BigSleep')
@click.option(           '--verbose', default=False, is_flag=True)
# input specification
@click.option(            '--prompt', default=None, type=str)
# output resolution
@click.option(     '--image_w', '-w', default=512, type=int)
@click.option(     '--image_h', '-h', default=512, type=int)
# model architectures
@click.option(         '--generator', default='biggan-deep-512')
@click.option(         '--perceptor', default=['ViT-B/32'], multiple=True)
@click.option(         '--p_weights', default=[1.], multiple=True)
# latent initialization
@click.option(       '--init_weight', default=None,  type=float)
@click.option(             '--image', default=None,  type=str)
# class-conditional parameters
@click.option(        '--class_loss', default=True)
@click.option(       '--max_classes', default=None,  type=int)
@click.option(     '--class_indices', default=None,  multiple=True)
@click.option( '--class_temperature', default=2e1)
# loss functions
@click.option(     '--lat_loss_coef', default=4e0)
@click.option(         '--loss_coef', default=1e2)
@click.option(   '--class_loss_coef', default=5e1)
@click.option(           '--tv_loss', default=0.,    type=float)
@click.option(         '--ssim_loss', default=0.,    type=float)
# training duration
@click.option(            '--epochs', default=1,     type=int)
@click.option(             '--steps', default=400,   type=int)
# optimization parameters
@click.option(        '--optimizer', default='Adam', type=str)
@click.option(                '--lr', default=7e-2,  type=float)
@click.option(         '--scheduler', default=None,  type=str)
@click.option(      '--weight_decay', default=0.,    type=float)
@click.option(         '--ema_decay', default=0.5,   type=float)
@click.option(   '--grad_accumulate', default=1,     type=int)
@click.option(      '--cutn_batches', default=1,     type=int)
# image subsampling
@click.option(              '--cutm', default=None,  type=str)
@click.option(              '--cutn', default=2**5,  type=int)
@click.option(              '--cutp', default=1.00,  type=float)
@click.option(          '--bilinear', default=False, is_flag=True)
@click.option(       '--center_bias', default=False, is_flag=True)
@click.option(      '--center_focus', default=2,     type=int)
@click.option('--experimental_resample', default=False, is_flag=True)
# data augmentation
@click.option(               '--aug', default=False, type=bool)
@click.option(         '--aug_noise', default=0.1,   type=float)
# output specification
@click.option(           '--folder', default=None)
@click.option(           '--bundle', default=False,  is_flag=True)
@click.option(       '--save_every', default=10)
@click.option(    '--save_progress', default=False,  is_flag=True)
@click.option(          '--preview', default=False,  is_flag=True)
# video compilation
@click.option(            '--video', default=False,  is_flag=True)
@click.option(            '--clean', default=False,  is_flag=True)
# device strategy
@click.option(           '--device', default='cuda:0')
@click.command()
@click.pass_context
def cli(ctx, seed, seeds, experiment, prompt, image, device, verbose, **kwargs):
    assert th.cuda.is_available(), 'ERROR <!> :: CUDA not available.'

    if seeds is None: seeds = [seed]
    else: seeds = [random.randint(0,1e6) for seed in range(seeds)]
    for seed in seeds:
        #---------------------------------------------------------------------------
        enforce_reproducibility(seed)
        #-----------------------------------------------------------------------
        outdir = os.getcwd() if kwargs['folder'] is None else kwargs['folder']
        os.makedirs(outdir, exist_ok=True)
        os.chdir(outdir)
        #-----------------------------------------------------------------------
        if os.path.isfile(prompt):
            with open(prompt, 'r') as f:
                prompts = f.readlines()
        else: prompts = [prompt]
        for prompt in prompts:
            print(f'`{prompt}` >> seed :: {seed} :: {experiment}')
            #-----------------------------------------------------------------------
            t2i = T2I(device=device, **kwargs).to(device)
            #-------------------------------------------------------------------
            t2i._generator(device=device, **kwargs)
            t2i._perceptor(device=device, **kwargs)
            t2i.optimizer ( t2i.init_z(), **kwargs)
            print(f'{kwargs["generator"]} :: {kwargs["perceptor"]}')
            #-----------------------------------------------------------------------
            run = neptune.init()
            run_id = run.get_url().split('/')[-1]
            #-----------------------------------------------------------------------
            for key,val in kwargs.items(): run[f'params/{key}'] = val
            config = config_prompt(prompt=prompt, seed=seed, step=kwargs['save_progress'])
            output_files = t2i.run(config, **kwargs)
            #-----------------------------------------------------------------------
            del t2i
            gc.collect()
            th.cuda.empty_cache()
            #-----------------------------------------------------------------------
    return output_files

if __name__ == '__main__':
    cli()
