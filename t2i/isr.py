import click, glob, os, pathlib, tqdm
import torch as th
from PIL import Image
from neurpy.model.pretrained import load

@click.option(  '--image', required=True)
@click.option(     '--up', default=4, type=int)
@click.option(  '--model', default='realesrgan', type=click.Choice(['realesrgan']))
@click.option( '--folder', default=os.getcwd())
@click.option( '--device', default='cuda:1')
@click.command()
@click.pass_context
def run(ctx, image, up, **args):

    if os.path.isdir(args['image']):
        files = glob.glob(os.path.join(args['image'], '*.png'))
    elif '*' in args['image']:
        files = glob.glob(args['image'])
    elif os.path.isfile(args['image']):
        files = [args['image']]
    elif isinstance(args['image'], list):
        files = args['image']
    else:
        raise Exception(f'invalid input `{args["image"]}`.')

    model = load(args['model'], image_size=(image_w, image_h))[0]

    output_files = []
    for filename in tqdm.tqdm(files):
        output_f = f'{pathlib.Path(filename).stem}-{up}x.png'

        if os.path.isfile(output_f): continue

        up_image = model.predict(Image.open(filename).convert('RGB'))

        print('{} -> {}'.format(filename, output_f))
        output_file = os.path.join(args['folder'], output_f)
        up_image.save(output_files)
        output_files.append(output_file)

    return output_files


if __name__ == '__main__':
    run()
