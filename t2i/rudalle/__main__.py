import click, gc, os, random, pprint, yaml
import neptune.new as neptune
from neptune.new.types import File
import torch as th
from t2i import config_prompt, isr
from t2i.rudalle import ruDALLE as T2I
from t2i.util import enforce_reproducibility


@click.option(              '--seed', default=random.randint(0,1e6))
@click.option(             '--seeds', default=1, type=int)
@click.option(        '--experiment', default='liberty3000/ruDALL-E')
@click.option(           '--verbose', default=False, is_flag=True)
# input specification
@click.option(            '--prompt', default=None, type=str)
# output resolution
@click.option(     '--image_w', '-w', default=256, type=int)
@click.option(     '--image_h', '-h', default=256, type=int)
@click.option(        '--batch_size', default=2**4)
# model architectures
@click.option(         '--generator', default='Malevich')
@click.option(         '--perceptor', default=['ruclip-vit-base-patch32-v5'], multiple=True)
@click.option(         '--p_weights', default=[1], multiple=True)
#-------------------------------------------------------------------------------
@click.option(               '--dwt', default=False, is_flag=True)
@click.option(              '--slow', default=False, is_flag=True)
@click.option(         '--use_cache', default=True)
@click.option(              '--slow', default=False, is_flag=True)
@click.option(             '--top_k', default=2048)
@click.option(             '--top_p', default=0.995)
@click.option(       '--temperature', default=1.0)
#-------------------------------------------------------------------------------
# output specification
@click.option(           '--folder', default=None)
@click.option(           '--bundle', default=False,  is_flag=True)
@click.option(       '--save_every', default=10)
@click.option(    '--save_progress', default=False,  is_flag=True)
@click.option(          '--preview', default=False,  is_flag=True)
@click.option(            '--isr', default=None, type=click.Choice([None,2,4,8]))
# video compilation
@click.option(            '--video', default=False,  is_flag=True)
@click.option(            '--clean', default=False,  is_flag=True)
# device strategy
@click.option(           '--device', default='cuda:0')
@click.command()
@click.pass_context
def cli(ctx, seed, seeds, experiment, prompt, device, verbose, **kwargs):
    assert th.cuda.is_available(), 'ERROR <!> :: CUDA not available.'

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
        print(f'{kwargs["generator"]}')
        #-----------------------------------------------------------------------
        run = neptune.init(project=experiment)
        run_id = run.get_url().split('/')[-1]
        #-----------------------------------------------------------------------
        for key,val in kwargs.items(): run[f'params/{key}'] = val
        config = config_prompt(prompt=prompt, seed=seed, step=kwargs['save_progress'])
        for key,val in config.items(): run[f'prompt/{key}'] = val

        output_files = t2i.run(config, seeds=seeds, **kwargs)
        for output_file in output_files: run['images'].log(File(output_file))
        #-----------------------------------------------------------------------
        del t2i
        gc.collect()
        th.cuda.empty_cache()
        #-----------------------------------------------------------------------
        if kwargs['isr'] is not None: ctx.invoke(isr.run, image=output_files)

    return output_files

if __name__ == '__main__':
    cli()
