import more_itertools, tqdm
import torch as th
import torch.nn.functional as F
from torchvision.utils import save_image
from deep_translator import GoogleTranslator
from transformers import top_k_top_p_filtering
from neurpy.model.pretrained import load
from t2i import T2I


class ruDALLE(T2I):
    def __init__(self, device='cuda:0', **kwargs):
        super().__init__()
        self.G, self.P = None, dict()
        self.device = device

    def _generator(self, generator, finetune=False, *args, **kwargs):
        bundle = load(generator, **kwargs)

        if finetune: self.dalle = finetune(**bundle, **kwargs['pretrain'])
        self.dalle = bundle['dalle']
        self.vae = bundle['vae']
        self.tokenizer = bundle['tokenizer']
        self.vocab_size = bundle['vocab_size']
        self.image_seq_length = bundle['image_seq_length']
        self.text_seq_length = bundle['text_seq_length']
        self.total_seq_length = bundle['total_seq_length']


    def translate(self, text, target='ru', source='auto'):
        return GoogleTranslator(src=source, target=target).translate(text)


    def attention_mask(self, bsize):
        shape = (bsize, 1, self.total_seq_length, self.total_seq_length)
        return th.tril(th.ones(shape, device=self.device))
        from rudalle import utils


    def vae_decode(self, output):
        codebooks = output[:, -self.image_seq_length:]
        images = self.vae.decode(codebooks)
        return images, codebooks


    def filter_logits(self, logits, top_k, top_p, temperature):
        logits = logits[:, -1, self.vocab_size:]
        logits /= temperature
        filtered_logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = th.nn.functional.softmax(filtered_logits, dim=-1)
        return probs


    def run(self, prompt_config, save_every=1, save_progress=False, device='cuda:0', **kwargs):

        prompt = prompt_config['prompt']

        self.ru = self.translate(prompt.lower().strip(), target='ru')
        self.en = self.translate(prompt.lower().strip(), target='en')
        self.input_ids = self.tokenizer.encode_text(self.ru, text_seq_length=self.text_seq_length)

        output_files = []
        if kwargs['slow'] or (kwargs['image_w'], kwargs['image_h']) not in [(256,256),(512,512)]:
            output_files = self.image_sequence(prompt, **kwargs)
        else:
            use_cache = kwargs['use_cache']
            n_chunk = kwargs['seeds'] // kwargs['batch_size']
            chunked = more_itertools.chunked(range(kwargs['seeds']), kwargs['batch_size'])
            chunks = tqdm.tqdm(enumerate(chunked), total=n_chunk, desc = '      epochs', position=0, leave=True)
            stepper = range(len(self.input_ids), self.total_seq_length)
            for i,chunk in chunks:
                chunk_bsize, has_cache = len(chunk), False
                #---------------------------------------------------------------
                with th.no_grad():
                    attention_mask = self.attention_mask(chunk_bsize)
                    output = self.input_ids.unsqueeze(0).repeat(chunk_bsize, 1).to(device)

                    for step in tqdm.tqdm(stepper, total=2**10, desc='   iteration', position=1, leave=True):
                        logits, has_cache = self.dalle(output[:,:step], attention_mask,
                        has_cache=has_cache, use_cache=use_cache, return_loss=False)

                        probs = th.multinomial(self.filter_logits(logits,
                        kwargs['top_k'], kwargs['top_p'], kwargs['temperature']), num_samples=1)
                        output = th.cat((output, probs), dim=-1)

                        if save_progress and steps % save_every == 0:
                            raise NotImplementedError
                    #-----------------------------------------------------------
                    images,codebooks = self.vae_decode(output)
                    for j,image in enumerate(images):
                        output_file = f'{prompt_config["seeded"]}.{(i+ 1) * (j + 1)}.png'
                        save_image(image, output_file)
                        output_files.append(output_file)

                    chunks.update(1)

        return output_files
