import click, gc, os, random, pprint, yaml
import neptune.new as neptune
import torch as th
from t2i import animate, config_prompt, isr
from t2i.vqlipse import VQLIPSE, refinement
from t2i.util import enforce_reproducibility


@click.option(              '--seed', default=random.randint(0,1e6))
@click.option(             '--seeds', default=None,  type=int)
@click.option(        '--experiment', default='liberty3000/VQLIPSE')
@click.option(           '--verbose', default=False, is_flag=True)
#-------------------------------------------------------------------------------
@click.option(            '--prompt', default=None, type=str)
#-------------------------------------------------------------------------------
@click.option(     '--image_w', '-w', default=320, type=int)
@click.option(     '--image_h', '-h', default=240, type=int)
#-------------------------------------------------------------------------------
@click.option(         '--generator', default='vqgan_imagenet_f16_16384')
@click.option(         '--perceptor', default=['ViT-B/32', 'ViT-B/16'], multiple=True)
@click.option(         '--p_weights', default=[1.,1.], multiple=True)
#-------------------------------------------------------------------------------
@click.option(              '--init', default=None, type=str)
@click.option(       '--init_weight', default=None, type=float)
#-------------------------------------------------------------------------------
@click.option(           '--tv_loss', default=0.,   type=float)
@click.option(         '--ssim_loss', default=0.,   type=float)
#-------------------------------------------------------------------------------
@click.option(            '--stages', default=\
'{"0":{"perceptor":["ViT-B/32","ViT-B/16"],\
  "cutn": 64, "cutp":0.5, "steps":300, "lr":0.10, "image_w":256, "image_h":256},\
  "1":{"perceptor":["ViT-B/32","ViT-B/16"],\
  "cutn": 48, "cutp":2.0, "steps":200, "lr":0.01, "image_w":512, "image_h":512},\
  "2":{"perceptor":["ViT-B/32","ViT-B/16"],\
  "cutn": 32, "cutp":1.0, "steps":100, "lr":0.01, "image_w":768, "image_h":768}}'
, type=str)
@click.option(         '--epochs', default=1,      type=int)
@click.option(          '--steps', default=500,    type=int)
#-------------------------------------------------------------------------------
@click.option(      '--optimizer', default='AdamW',type=str)
@click.option(             '--lr', default=1e-1,   type=float)
@click.option(      '--scheduler', default=None,   type=str)
@click.option(   '--weight_decay', default=1e-5,   type=float)
@click.option(      '--ema_decay', default=None,   type=float)
@click.option('--grad_accumulate', default=1,      type=int)
@click.option(   '--cutn_batches', default=1,      type=int)
#-------------------------------------------------------------------------------
@click.option(           '--cutm', default='v2',   type=str)
@click.option(           '--cutn', default=2**5,   type=int)
@click.option(           '--cutp', default=0.50,   type=float)
#-------------------------------------------------------------------------------
@click.option(            '--aug', default=True,   type=bool)
@click.option(      '--aug_noise', default=0.1,    type=float)
#-------------------------------------------------------------------------------
@click.option(         '--folder', default=None)
@click.option(         '--bundle', default=False,  is_flag=True)
@click.option(     '--save_every', default=10)
@click.option(  '--save_progress', default=False,  is_flag=True)
@click.option(        '--preview', default=False,  is_flag=True)
@click.option(            '--isr', default=None, type=click.Choice([None,2,4,8]))
#-------------------------------------------------------------------------------
@click.option(      '--zoom_init', default=200,    type=int)
@click.option(      '--zoom_step', default=10,     type=int)
@click.option(        '--zoom_2d', default=False,  is_flag=True)
@click.option(      '--zoom_rate', default=25e-3,  type=float)
@click.option(        '--zoom_3d', default=False,  is_flag=True)
@click.option(    '--border_mode', default='reflection',
type=click.Choice(['border','reflection','zeros']))
@click.option(  '--sampling_mode', default='bicubic',
type=click.Choice(['bicubic','bilinear','nearest']))
@click.option(    '--lock_camera', default=True,   type=bool)
@click.option(  '--field_of_view', default=40,     type=int)
@click.option(     '--near_plane', default=1,      type=int)
@click.option(      '--far_plane', default=10_000, type=int)
@click.option(    '--translate_x', default= '-10 * cos(t)')
@click.option(    '--translate_y', default= '10 * sin(t)')
@click.option(    '--translate_z', default= '75', help='only used if `animate` == `3D`.')
@click.option(      '--rotate_3d', default='[1,0,0,.01]',\
help='must be a [w,x,y,z] rotation (unit) quaternion. use `--rotate_3d=[1,0,0,0]` for no rotation.')
@click.option(      '--stabilize', default=False)
#-------------------------------------------------------------------------------
@click.option(          '--video', default=False,  is_flag=True)
@click.option(            '--fps', default=30,     type=int)
@click.option(          '--clean', default=False,  is_flag=True)
#-------------------------------------------------------------------------------
@click.option(         '--device', default='cuda:0')
@click.command()
@click.pass_context
def cli(ctx, seed, seeds, experiment, prompt, init, device, video, **kwargs):
    assert th.cuda.is_available(), 'ERROR <!> :: CUDA not available.'

    for seed in [seed] if seeds is None else [random.randint(0,1e6) for seed in range(seeds)]:
        #-----------------------------------------------------------------------
        enforce_reproducibility(seed)
        #-----------------------------------------------------------------------
        outdir = os.getcwd() if kwargs['folder'] is None else kwargs['folder']
        os.makedirs(outdir, exist_ok=True)
        os.chdir(outdir)
        #-----------------------------------------------------------------------
        if os.path.isfile(prompt) and prompt.endswith('.txt'):
            with open(prompt, 'r') as f: prompts = f.readlines()
        else: prompts = [prompt]
        for prompt in prompts:
            print(f'`{prompt}` >> seed :: {seed} :: {experiment}')
            #-------------------------------------------------------------------
            if init == 'bigsleep':
                from t2i.bigsleep.__main__ import cli
                output_files = ctx.invoke(cli, seed=seed, prompt=prompt, steps=250)
                init = output_files[-1]
            if init =='latent_diffusion':
                from t2i.latent_diffusion.__main__ import cli
                output_files = ctx.invoke(cli, seed=seed, prompt=prompt, batch_size=1)
                init = output_files[-1]
            elif init == 'rudalle':
                from t2i.rudalle.__main__ import cli

                image_w = 512 if kwargs['image_w'] * kwargs['image_h'] >= (256 * 256) else 256
                image_h = 512 if kwargs['image_w'] * kwargs['image_h'] >= (256 * 256) else 256
                output_files = ctx.invoke(cli, image_w=image_w, image_h=image_h,
                                          seed=seed, prompt=prompt, batch_size=1)
                init = output_files[-1]
                prompt += f'| {init}'

            if init == 'perlin': args = dict(octaves=2**2, weight=22e-2, **kwargs)
            elif init == 'pyramid': args = dict(octaves=2**3,  decay=99e-2, **kwargs)
            else: args = kwargs

            _init = init
            #-------------------------------------------------------------------
            vqlipse = VQLIPSE(**args).to(device)
            #-------------------------------------------------------------------
            run = neptune.init(project=experiment)
            run_id = run.get_url().split('/')[-1]
            run['seed'] = seed
            #-------------------------------------------------------------------
            outputs = []
            for i,args in enumerate(refinement(**args)):
                for key,val in args.items(): run[f'params/stage_{i+1}/{key}'] = val
                config = config_prompt(prompt=prompt, seed=seed, step=args['save_progress'])
                for key,val in config.items(): run[f'prompt/{key}'] = val
                #---------------------------------------------------------------
                vqlipse._generator(device=device, **args)
                vqlipse._perceptor(device=device, **args)
                print(f'{args["generator"]} :: {args["perceptor"]}')
                #---------------------------------------------------------------
                vqlipse.optimizer(vqlipse.init_z(init=_init, **args), **args)
                #---------------------------------------------------------------
                output_files = vqlipse.run(config, **args)
                _init = output_files[-1]
                #---------------------------------------------------------------
                del vqlipse.z
                del vqlipse.z_init
            #-------------------------------------------------------------------
            del vqlipse
            gc.collect()
            th.cuda.empty_cache()
            #-------------------------------------------------------------------
            if kwargs['isr'] is not None:
                output_files = ctx.invoke(isr.run, output_files, kwargs['isr'])
            if video:
                animate.video(output_files, config['video'], **kwargs)
            #-------------------------------------------------------------------
        #-----------------------------------------------------------------------
    return output_files


if __name__ == '__main__':
    cli()
