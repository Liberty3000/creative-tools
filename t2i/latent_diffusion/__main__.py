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
'{"stages":null, "image_w":768, "image_h":768, "lr":0.01, "steps":100, "cutn":32, "tv_loss":1.0, \
  "cutp":1.0, "cutn_batches":4, "init_weight":5.0, "aug":false, "perceptor": ["ViT-B/32", "RN50x4"]}',
type=str)
@click.option(          '--isr', default=None,   type=click.Choice([None,'2','4','8']))
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
    output_prompts = []
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
        for _ in range(len(output_files)): output_prompts.append(prompt)
        for output_file in output_files: run['images'].log(File(output_file))
    #---------------------------------------------------------------------------
    del t2i
    gc.collect()
    th.cuda.empty_cache()
    #---------------------------------------------------------------------------
    if kwargs['isr'] is not None:
        ctx.invoke(isr.run, input=output_files)
    if kwargs['refine'] is not None:
        from t2i.vqlipse.__main__ import cli
        for prompt, output_file in zip(output_prompts, output_files):
            if kwargs['refine_prompt']: prompt += f'|{output_file}'
            args = json.loads(kwargs['refine'])
            ctx.invoke(cli, prompt=prompt, init=output_file, **args)
    #---------------------------------------------------------------------------
    return output_files


if __name__ == '__main__':
    cli()
