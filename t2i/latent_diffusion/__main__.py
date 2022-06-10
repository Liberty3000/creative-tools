import click, gc, json, os, random, pprint, yaml
import neptune.new as neptune
from neptune.new.types import File
import torch as th
from t2i import config_prompt, isr
from t2i.latent_diffusion import LatentDiffusion as T2I
from t2i.util import enforce_reproducibility


@click.option(         '--seed', default=random.randint(0,1e6))
@click.option(        '--seeds', default=1, type=int)
@click.option(   '--experiment', default='liberty3000/Latent-Diffusion')
@click.option(      '--verbose', default=False, is_flag=True)
#-------------------------------------------------------------------------------
@click.option(       '--prompt', default=None, type=str)
#-------------------------------------------------------------------------------
@click.option('--image_w', '-w', default=256, type=int)
@click.option('--image_h', '-h', default=256, type=int)
@click.option(   '--batch_size', default=2**2)
#-------------------------------------------------------------------------------
@click.option(    '--generator', default='text2img-large')
#-------------------------------------------------------------------------------
@click.option(   '--ddim_steps', default=200)
@click.option(     '--ddim_eta', default=0.0)
@click.option(       '--n_iter', default=1)
@click.option(        '--scale', default=5)
@click.option(         '--plms', default=True)
#-------------------------------------------------------------------------------
@click.option(       '--folder', default=None)
@click.option(       '--bundle', default=False,  is_flag=True)
@click.option(   '--save_every', default=10)
@click.option('--save_progress', default=False,  is_flag=True)
@click.option(      '--preview', default=False,  is_flag=True)
@click.option('--refine_prompt', default=False,  is_flag=True)
@click.option(       '--refine', default=\
'{"mode":"vqlipse", "stages":null, "image_w":768, "image_h":768, "lr":0.01, "steps":10, "ema_decay":0.985, \
  "cutn":16, "cutn_batches":16, "tv_loss":10, "init_weight":2000.0, "perceptor": ["ViT-B/32", "ViT-B/16", "RN50", "RN50x4"]}',\
  type=str)
@click.option(          '--isr', default=2,   type=click.Choice([None,'2','4','8']))
#-------------------------------------------------------------------------------
@click.option(       '--device', default='cuda:0')
@click.command()
@click.pass_context
def cli(ctx, seed, seeds, experiment, prompt, device, verbose, **kwargs):
    assert th.cuda.is_available(), 'ERROR <!> :: CUDA not available.'
    #---------------------------------------------------------------------------
    enforce_reproducibility(seed)
    #---------------------------------------------------------------------------
    outdir = os.getcwd() if kwargs['folder'] is None else kwargs['folder']
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    #---------------------------------------------------------------------------
    t2i = T2I(device=device, **kwargs)
    #---------------------------------------------------------------------------
    t2i._generator(device=device, **kwargs)
    print(f'{kwargs["generator"]}')
    #---------------------------------------------------------------------------
    if os.path.isfile(prompt):
        with open(prompt, 'r') as f:
            prompts = f.readlines()
    else: prompts = [prompt]
    #---------------------------------------------------------------------------
    outputs = []
    for prompt in prompts:
        print(f'`{prompt}` >> seed :: {seed} :: {experiment}')
        #-----------------------------------------------------------------------
        run = neptune.init(project=experiment)
        run_id = run.get_url().split('/')[-1]
        run['seed'] = seed
        #-----------------------------------------------------------------------
        for key,val in kwargs.items(): run[f'params/{key}'] = val
        config = config_prompt(prompt=prompt, seed=seed, step=kwargs['save_progress'])
        for key,val in config.items(): run[f'prompt/{key}'] = val

        output_files = t2i.run(config, seeds=seeds, **kwargs)
        for output_file in output_files: run['images'].log(File(output_file))
        if kwargs['isr'] is not None:
            output_files = ctx.invoke(isr.run, up=kwargs['isr'], input=output_files)
        outputs += [(prompt,f) for f in output_files]
    #---------------------------------------------------------------------------
    del t2i
    gc.collect()
    th.cuda.empty_cache()
    #---------------------------------------------------------------------------
    if kwargs['refine'] is not None:
        args = json.loads(kwargs['refine'])
        mode = args.pop('mode')
        if mode == 'guided_diffusion':
            from t2i.guided_diffusion.__main__ import cli
        if mode == 'vqlipse':
            from t2i.vqlipse.__main__ import cli
        for prompt, output_file in outputs:
            prompt += f'|{output_file}'
            ctx.invoke(cli, prompt=prompt, init=output_file, **args)
    #---------------------------------------------------------------------------
    return output_files


if __name__ == '__main__':
    cli()
