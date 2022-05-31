import click, glob, os, pathlib, tqdm
import torch as th
from PIL import Image
from neurpy.model.pretrained import load

@click.option(  '--input', required=True)
@click.option(     '--up', default=4, type=int)
@click.option(  '--model', default='realesrgan', type=click.Choice(['realesrgan']))
@click.option( '--folder', default=os.getcwd())
@click.option( '--device', default='cuda:1')
@click.command()
@click.pass_context
def run(ctx, input, up, **args):

    if os.path.isdir(input):
        files = glob.glob(os.path.join(input, '*.png'))
    elif '*' in input:
        files = glob.glob(input)
    elif os.path.isfile(input):
        files = [input]
    elif isinstance(input, list):
        files = input
    else:
        raise Exception(f'invalid input `{args["input"]}`.')

    model = load(args['model'], up=up, device=args['device'])[0]

    output_files = []
    bar = tqdm.tqdm(files, total=len(files))
    for filename in bar:
        try:
            output_f = f'{pathlib.Path(filename).stem}-x{up}.png'

            if os.path.isfile(output_f): continue

            image = Image.open(filename).convert('RGB')
            up_input = model.predict(image)

            bar.set_description('{} -> {}'.format(filename[:4], output_f[:4]))
            output_file = os.path.join(args['folder'], output_f)
            up_input.save(output_file)
            output_files.append(output_file)
        except: pass

    return output_files


if __name__ == '__main__':
    run()
