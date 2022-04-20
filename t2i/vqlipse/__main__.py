import click, gc, os, random, pprint, yaml
import neptune.new as neptune
import torch as th
from t2i import config_prompt
from t2i.vqlipse import VQLIPSE, refinement
from t2i.util import enforce_reproducibility


@click.option(              '--seed', default=random.randint(0,1e6))
@click.option(        '--experiment', default='liberty3000/VQLIPSE')
@click.option(           '--verbose', default=False, is_flag=True)
# input specification
@click.option(            '--prompt', default=None, type=str)
# output resolution
@click.option(     '--image_w', '-w', default=224, type=int)
@click.option(     '--image_h', '-h', default=224, type=int)
# model architectures
@click.option(         '--generator', default='vqgan_imagenet_f16_16384')
@click.option(         '--perceptor', default=['ViT-B/32'], multiple=True)
@click.option(         '--p_weights', default=[1.], multiple=True)
# latent initialization
@click.option(       '--init_weight', default=None,  type=float)
@click.option(             '--image', default=None,  type=str)
@click.option(            '--perlin', default=None,  is_flag=True)
@click.option(           '--pyramid', default=None,  is_flag=True)
@click.option(              '--init', default=None,  type=click.Choice([None,'rudalle','bigsleep']))
# loss functions
@click.option(           '--tv_loss', default=0.,    type=float)
@click.option(         '--ssim_loss', default=0.,    type=float)
# training duration
@click.option(            '--stages', default=\
'{"0":{"perceptor":["ViT-B/32","ViT-B/16","RN50"], "p_weights":[1.0,1.0,1.0],\
  "cutn":64, "cutp":0.50, "steps":150, "init_weight":0.0, "lr":0.10, "image_w":320, "image_h":240},\
  "1":{"perceptor":["ViT-B/32","ViT-B/16","RN50"], "p_weights":[1.0,1.0,1.0],\
  "cutn":32, "cutp":1.00, "steps":150, "init_weight":2.0, "lr":0.01, "image_w":640, "image_h":480},\
  "2":{"perceptor":["ViT-B/32","ViT-B/16","RN50"], "p_weights":[1.0,1.0,1.0],\
  "cutn":16, "cutp":1.00, "steps":150, "init_weight":2.0, "lr":0.01, "image_w":1024,"image_h":768}}',\
type=str)
@click.option(         '--epochs', default=1,     type=int)
@click.option(          '--steps', default=200,   type=int)
# optimization parameters
@click.option(      '--optimizer', default='AdamW',type=str)
@click.option(             '--lr', default=1e-1,  type=float)
@click.option(      '--scheduler', default=None,  type=str)
@click.option(   '--weight_decay', default=1e-5,  type=float)
@click.option(      '--ema_decay', default=None,  type=float)
@click.option('--grad_accumulate', default=1,     type=int)
@click.option(   '--cutn_batches', default=1,     type=int)
# image subsampling
@click.option(           '--cutm', default='v2',  type=str)
@click.option(           '--cutn', default=2**5,  type=int)
@click.option(           '--cutp', default=0.50,  type=float)
# data augmentation
@click.option(            '--aug', default=True,  type=bool)
@click.option(      '--aug_noise', default=0.1,   type=float)
# output specification
@click.option(         '--folder', default=None)
@click.option(         '--bundle', default=False,  is_flag=True)
@click.option(     '--save_every', default=10)
@click.option(  '--save_progress', default=False,  is_flag=True)
@click.option(        '--preview', default=False,  is_flag=True)
# animation transformations
@click.option(           '--zoom', default=False,  is_flag=True)
@click.option(          '--swirl', default=False,  is_flag=True)
@click.option(          '--depth', default=False,  is_flag=True)
# video compilation
@click.option(          '--video', default=False,  is_flag=True)
@click.option(          '--clean', default=False,  is_flag=True)
# device strategy
@click.option(         '--device', default='cuda:0')
@click.command()
@click.pass_context
def cli(ctx, seed, experiment, prompt, image, device, verbose, **kwargs):
    assert th.cuda.is_available(), 'ERROR <!> :: CUDA not available.'
    #---------------------------------------------------------------------------
    enforce_reproducibility(seed)
    #---------------------------------------------------------------------------
    if kwargs['perlin']:  kwargs['perlin'] = dict(octaves=2**2, weight=22e-2)
    if kwargs['pyramid']: kwargs['pyramid']= dict(octaves=2**3, decay=99e-2)
    #-----------------------------------------------------------------------
    outdir = os.getcwd() if kwargs['folder'] is None else kwargs['folder']
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    #-----------------------------------------------------------------------
    if os.path.isfile(prompt):
        with open(prompt, 'r') as f: prompts = f.readlines()
    else: prompts = [prompt]
    for prompt in prompts:
        print('`{}` >> seed :: {} :: VQLIPSE'.format(prompt, seed))
        #-----------------------------------------------------------------------
        if kwargs['init'] == 'bigsleep':
            from t2i.bigsleep.__main__ import cli
            output_files = ctx.invoke(cli, seed=seed, prompt=prompt,
            image_w=kwargs['image_w'], image_h=kwargs['image_h'])
            image = output_files[-1]
        if kwargs['init'] == 'rudalle':
            from t2i.rudalle.__main__ import cli
            output_files = ctx.invoke(cli, seed=seed, prompt=prompt,
            image_w=kwargs['image_w'], image_h=kwargs['image_h'])
            image = output_files[-1]
        #-----------------------------------------------------------------------
        vqlipse = VQLIPSE(**kwargs).to(device)
        #-----------------------------------------------------------------------
        run = neptune.init()
        run_id = run.get_url().split('/')[-1]
        #-----------------------------------------------------------------------
        outputs = []
        for i,args in enumerate(refinement(**kwargs)):
            for key,val in args.items(): run[f'params/stage_{i+1}/{key}'] = val
            config = config_prompt(prompt=prompt, seed=seed, step=args['save_progress'])
            #-------------------------------------------------------------------
            vqlipse._generator(device=device, **args)
            vqlipse._perceptor(device=device, **args)
            print(f'{args["generator"]} :: {args["perceptor"]}')
            #-------------------------------------------------------------------
            init_args = dict(image=image, image_w=args['image_w'],
            image_h=args['image_h'], perlin=args['perlin'], pyramid=args['pyramid'])

            optm_args = dict(optimizer=args['optimizer'], lr=args['lr'],
            ema_decay=args['ema_decay'], weight_decay=args['weight_decay'])

            vqlipse.optimizer(vqlipse.init_z(**init_args), **optm_args)
            #-------------------------------------------------------------------
            output_files = vqlipse.run(config, **args)
            outputs.append(output_files)
            image = output_files[-1]
            #-------------------------------------------------------------------
            del vqlipse.z
            del vqlipse.z_init
        #-----------------------------------------------------------------------
        del vqlipse
        gc.collect()
        th.cuda.empty_cache()
    return outputs


if __name__ == '__main__':
    cli()
