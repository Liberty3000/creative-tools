import click, gc, json, os, random, pprint, yaml
import neptune.new as neptune
from neptune.new.types import File
import torch as th
from t2i.guided_diffusion import GuidedDiffusion as T2I
from t2i.util import enforce_reproducibility
from t2i import animate, config_prompt, from_config, isr


@click.option(              '--seed', default=random.randint(0,1e6))
@click.option(             '--seeds', default=1, type=int)
@click.option(        '--experiment', default='liberty3000/Guided-Diffusion')
@click.option(           '--verbose', default=False, is_flag=True)
#-------------------------------------------------------------------------------
@click.option(            '--prompt', default=None, type=str)
#-------------------------------------------------------------------------------
@click.option(     '--image_w', '-w', default=512, type=int)
@click.option(     '--image_h', '-h', default=512, type=int)
@click.option(        '--batch_size', default=1)
#-------------------------------------------------------------------------------
@click.option(         '--generator', default='512x512_diffusion_uncond_finetune_008100')
@click.option(         '--perceptor', default=['ViT-B/32', 'ViT-B/16', 'RN50'], multiple=True)
@click.option(         '--p_weights', default=[1.,1.,1.], multiple=True)
#-------------------------------------------------------------------------------
@click.option(              '--init', default=None, type=str)
@click.option(       '--init_weight', default=1000, type=float)
#-------------------------------------------------------------------------------
@click.option(     '--guidance_loss', default=1500)
@click.option(           '--tv_loss', default=150)
@click.option(        '--range_loss', default=100)
@click.option(          '--sat_loss', default=0)
@click.option(       '--lpips_model', default='vgg')
#-------------------------------------------------------------------------------
@click.option(            '--epochs', default=1,      type=int)
@click.option(             '--steps', default=400,    type=str)
@click.option(   '--diffusion_steps', default=1000)
@click.option(    '--skip_timesteps', default=0)
#-------------------------------------------------------------------------------
@click.option(              '--cutm', default='v4')
@click.option(              '--cutn', default=32)
@click.option(              '--cutp', default=1.00,   type=float)
@click.option(      '--cutn_batches', default=4)
#-------------------------------------------------------------------------------
@click.option(               '--aug', default=True,   type=bool)
#-------------------------------------------------------------------------------
@click.option(        '--clamp_grad', default=True)
@click.option(     '--clip_denoised', default=False)
@click.option(   '--randomize_class', default=True)
@click.option(          '--ddim_eta', default=0.5)
@click.option(      '--fuzzy_prompt', default=False)
@click.option(          '--rand_mag', default=0.05)
#-------------------------------------------------------------------------------
@click.option(           '--preview', default=False,  is_flag=True)
@click.option(            '--folder', default=os.getcwd())
@click.option(            '--bundle', default=False,  is_flag=True)
#-------------------------------------------------------------------------------
@click.option(        '--save_every', default=10)
@click.option(     '--save_progress', default=False,  is_flag=True)
@click.option(             '--video', default=False,  is_flag=True)
@click.option(               '--fps', default=30,     type=int)
@click.option(             '--clean', default=False,  is_flag=True)
@click.option(               '--isr', default=None,   type=click.Choice([None,'2','4','8']))
#-------------------------------------------------------------------------------
@click.option(       '--refine', default=\
'{"mode":"vqlipse", "stages":null, "image_w":768, "image_h":768, "steps":200, "lr":0.01, \
  "cutm":"v4", "cutn":48, "tv_loss": 10.0, "ema_decay":0.99, "cutp":1.0, "cutn_batches":4, "aug":false, \
  "init_weight":100.0, "perceptor": ["ViT-B/32", "ViT-B/16", "RN50"]}',\
  type=str)
@click.option('--refine_image_prompt', default=True)
#-------------------------------------------------------------------------------
@click.option(            '--device', default='cuda:0')
@click.command()
@click.pass_context
def cli(ctx, seed, seeds, experiment, prompt, device, video, **kwargs):
    assert th.cuda.is_available(), 'ERROR <!> :: CUDA not available.'
    #-----------------------------------------------------------------------
    outdir = os.getcwd() if kwargs['folder'] is None else kwargs['folder']
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    #---------------------------------------------------------------------------
    t2i = T2I(device=device, **kwargs)
    t2i._generator(device=device, **kwargs)
    t2i._perceptor(device=device, **kwargs)
    print(f'{kwargs["generator"]} :: {kwargs["perceptor"]}')
    #-----------------------------------------------------------------------
    for seed in [seed] if seeds is None else [random.randint(0,1e6) for seed in range(seeds)]:
        #-----------------------------------------------------------------------
        enforce_reproducibility(seed)
        #-----------------------------------------------------------------------
        if os.path.isfile(prompt) and prompt.endswith('.txt'):
            with open(prompt, 'r') as f: prompts = f.readlines()
        else: prompts = [prompt]
        for prompt in prompts:
            #-------------------------------------------------------------------
            print(f'`{prompt}` >> seed :: {seed} :: {experiment}')
            #-------------------------------------------------------------------
            run = neptune.init(project=experiment)
            run_id = run.get_url().split('/')[-1]
            run['seed'] = seed
            #-------------------------------------------------------------------
            config = config_prompt(prompt=prompt, seed=seed, step=kwargs['save_progress'])
            for key,val in config.items(): run[f'prompt/{key}'] = val
            for key,val in kwargs.items(): run[f'params/{key}'] = val

            output_files = t2i.run(config, seeds=seeds, **kwargs)
            for output_file in output_files: run['images'].log(File(output_file))
            #-------------------------------------------------------------------
            if kwargs['isr'] is not None:
                output_files = ctx.invoke(isr.run, input=output_files)
            if kwargs['refine'] is not None:
                from t2i.vqlipse.__main__ import cli
                if kwargs['refine_image_prompt']: prompt += f'|{output_files[-1]}'
                args = from_config(kwargs['refine'])
                init = os.path.join(outdir, output_files[-1])
                ctx.invoke(cli, prompt=prompt, init=init, **args)

            if video: animate.video(output_files, config['video'], **kwargs)
            #-------------------------------------------------------------------
    #---------------------------------------------------------------------------
    return output_files

if __name__ == '__main__':
    cli()
