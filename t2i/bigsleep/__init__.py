import random
import torch as th
import torch.nn.functional as F
from neurpy.model.pretrained import load
from neurpy.module.ema import EMA
from t2i import T2I

def init_class_fewhot(latent=None, num_classes=1000, fewhot=[]):
    if not latent: latent = th.zeros(num_classes)
    for idx in fewhot: latent[idx] = 1
    return latent


def rand_cutout(image, size, center_bias=False, center_focus=2, cutp=1, **kwargs):
    sideY, sideX = image.shape[2:4]
    max_size = min(sideX, sideY)
    min_size = min(sideX, sideY, size)

    if center_bias:
        center = sideY / 2
        offset_x = int(random.gauss(mu=center, sigma=center / center_focus))
        offset_x = random.randint(0, max_offset) if (offset_x > max_offset or offset_x < min_offset) else offset_x

        center = sideX / 2
        offset_y = int(random.gauss(mu=center, sigma=center / center_focus))
        offset_y = random.randint(0, max_offset) if (offset_y > max_offset or offset_y < min_offset) else offset_y
    else:

        size = int(th.rand([])**cutp * (max_size - min_size) + min_size)
        offset_x = th.randint(0, sideY - size + 1, ())
        offset_y = th.randint(0, sideX - size + 1, ())

    cutout = image[:, :, offset_x:offset_x + size, offset_y:offset_y + size]
    return cutout


def differentiable_topk(x, k, temperature=1.):
    n, dim = x.shape
    topk_tensors = []

    for i in range(k):
        is_last = i == (k - 1)
        values, indices = (x / temperature).softmax(dim=-1).topk(1, dim=-1)
        topks = th.zeros_like(x).scatter_(-1, indices, values)
        topk_tensors.append(topks)
        if not is_last: x = x.scatter(-1, indices, float('-inf'))

    topks = th.cat(topk_tensors, dim=-1)
    return topks.reshape(n, k, dim).sum(dim = 1)


class Latent(th.nn.Module):
    def __init__(
        self,
        latent_shape=(1,512),
        num_classes = 1000,
        max_classes = None,
        class_temperature = 2.,
        std=1,

        class_indices=[],
        clip=False,
        clip_range=(-6,6)
    ):
        super().__init__()

        assert not max_classes is not None or max_classes > 0 and max_classes <= num_classes,\
        f'max_classes must be between 0 and {num_classes}'
        self.max_classes, self.class_temperature = max_classes, class_temperature
        self.class_indices = class_indices
        self.clip, self.clip_range = clip, clip_range

        latents = th.zeros(*latent_shape).normal_(std=std)

        if not class_indices:
            classes = th.zeros(latent_shape[0], num_classes).normal_(mean=-3.9,std=0.3)
        else:
            classes = th.zeros(num_classes)
            classes[class_indices] = 1
            classes = classes.repeat(latent_shape[0], 1)

        self.distr = th.nn.Parameter(latents)
        self.categ = th.nn.Parameter(classes)
        self.register_buffer('thresh_lat', th.tensor(1))

    def forward(self):
        if self.max_classes is not None:
            classes = differentiable_topk(self.categ, self.max_classes,
                                          temperature=self.class_temperature)
        else:
            if self.class_indices: classes = th.softmax(self.categ, dim=-1)
            else: classes = th.sigmoid(self.categ)

        distr = self.distr
        if self.clip: distr = distr.clip(*self.clip_range)
        return distr, classes


def similarity_score(text_embed, image_embed, text_type='max', coef=1):
    sim = th.cosine_similarity(text_embed, image_embed, dim=-1).mean()
    return {'max':-1,'min':1}[text_type] * sim


class BigSleep(T2I):
    def __init__(self, device='cuda:0', **kwargs):
        super().__init__()
        self.G, self.P = None, dict()
        self.device = device

    def _generator(self, generator, device='cuda', *args, **kwargs):
        self.G,*_ = load(generator)
        self.z_dim = (len(self.G.config.layers) + 1, self.G.config.z_dim)
        self.num_classes = self.G.config.num_classes
        return self.G.to(device)


    def encode_text(self, raw):
        self.prompts = []
        for p,p_dict in self.P.items():
            tokenize = p_dict['tokenize']
            perceptor= p_dict['perceptor']
            for prompt in raw.split('|'):
                with th.no_grad():
                    self.prompts += [perceptor.encode_text(tokenize(prompt).to(self.device)).detach()]

    def init_z(self, ema_decay=None, device='cuda:0', **kwargs):
        z = Latent(latent_shape=self.z_dim, num_classes=self.num_classes, **kwargs).to(device)
        return z


    def optimizer(self, z, lr, ema_decay=None, optimizer='Adam', **kwargs):
        self.z = z
        Opt = getattr(th.optim, optimizer)
        if ema_decay is not None: z = EMA(self.z, ema_decay)
        self.optim = Opt(self.z.parameters(), lr=lr, weight_decay=kwargs['weight_decay'])
    #---------------------------------------------------------------------------

    def encode_text(self, text, normalize=False, verbose=False):
        self.text = []
        for p, p_dict in self.P.items():
            perceptor, tokenize = p_dict['perceptor'], p_dict['tokenize']
            with th.no_grad():
                inputs = tokenize(text).to(self.device)
                text_encoding = perceptor.encode_text(inputs)
                self.text.append(text_encoding.detach())

    def encode_image(self, image):
        p = list(self.P.values())[0]
        perceptor = p['perceptor']
        if isinstance(image, str): image = Image.open(image)
        normed_img = self.image_transform(image)
        normed_img = normed_img.unsqueeze(0).to(self.device)
        with th.no_grad():
            encoding = perceptor.encode_image(normed_img).detach()
        return encoding


    def generate(self):
        self.G.eval()
        output = self.G(*self.z(), 1)
        output = (output + 1) / 2
        return output


    def training_step(self, cutn_batches=1, grad_accumulate=1, *args, **kwargs):
        self.z.train()
        total_loss=0.
        for cutn_batch in range(cutn_batches):
            self.optim.zero_grad(set_to_none=True)
            for grad_step in range(grad_accumulate):
                outputs = self.forward(*args, **kwargs)
                loss = sum(outputs['losses']) / grad_accumulate
                total_loss += loss.item()
                loss.backward()
            self.optim.step()
            if hasattr(self.z, 'update'): self.z.update()
            self.optim.zero_grad()
        return dict(**outputs, loss=loss.item(), total_loss=total_loss)


    def forward(self, loss_coef, class_loss, class_loss_coef, **kwargs):
        image = self.generate()
        #-----------------------------------------------------------------------
        latents, soft_onehots = self.z()
        losses = []
        #-----------------------------------------------------------------------
        cls_loss = 0 if not class_loss else self.class_loss(soft_onehots, **kwargs)
        #-----------------------------------------------------------------------
        for perceptor,p_dict in self.P.items():
            perceptor = p_dict['perceptor']
            tokenize = p_dict['tokenize']
            weight = p_dict['weight']

            patches = self.make_cutouts(image, patch_size=p_dict['patch_size'], **kwargs)
            image_embed = perceptor.encode_image(patches)

            for text_embed in self.text:
                sim = similarity_score(text_embed, image_embed, 'max', loss_coef)
                losses.append(weight * sim)

        sim_loss = sum(losses).mean()
        #-----------------------------------------------------------------------
        lat_loss = self.latent_loss(latents)
        for array in latents:
            diffs = array - th.mean(array)
            var = th.mean(th.pow(diffs, 2.0))
            std = th.pow(var, 0.5)
            z_scores = diffs / std
            skews = th.mean(th.pow(z_scores, 3.0))
            kurtoses = th.mean(th.pow(z_scores, 4.0)) - 3.0
            lat_loss += th.abs(kurtoses) / latents.shape[0] + th.abs(skews) / latents.shape[0]
        #-----------------------------------------------------------------------
        return dict(image=image, losses=losses, similarity_loss=sim_loss,
                    latent_loss=lat_loss, class_loss=cls_loss)

    def latent_loss(self, latents, lat_loss_coef=4, *args, **kwargs):
        mean_ = th.abs(th.mean(latents, dim=1)).mean()
        stdv_ = th.abs(1 - th.std(latents, dim=1)).mean()
        mmse_ = th.max(th.square(latents).mean(), self.z.thresh_lat)
        lat_loss = mean_ + stdv_ + lat_loss_coef * mmse_
        return lat_loss

    def class_loss(self, soft_onehots, class_loss_coef=1., *args, **kwargs):
        topk = th.topk(soft_onehots, largest=False, dim=1, k=999)[0]
        return ((class_loss_coef * topk) **2).mean()

    def input_modalities(self, prompt=None, image=None, latent=None):
        encodings = []
        if text is not None:
            for prompt_min in text.split(text_delimiter):
                if encoding is not None:
                    encoding = encoding.to(self.device)
                    encodings.append(encoding)
                elif text is not None and image is not None:
                    encoding = self.encode_text(text) + self.encode_image(image)
                    encoding /= 2
                    encodings.append(encoding)
                elif text is not None:
                    encodings.append(self.encode_text(text))
                elif image is not None:
                    for k,v in self.P.items():
                        encodings.append(encoding)
        else:
            if encoding is not None: encoding = encoding.to(self.device)
            elif image is not None: encoding = self.encode_image(image)
            encodings = [encoding]
        return encodings

    def make_cutouts(self, output, cutn, patch_size, image_w, aug=True, aug_noise=0.1, experimental_resample=False, **kwargs):
        pieces = []
        interp_args = dict()
        for ch in range(cutn):
            size = int(image_w * th.zeros(1,).normal_(mean=.8, std=.3).clip(.5, .95))
            patch = rand_cutout(output, size, **kwargs)
            if experimental_resample:
                patch = resample(patch, patch_size)
            else:
                patch = F.interpolate(patch, patch_size, **interp_args)
            pieces.append(patch)
        patches = th.cat(pieces).to(self.device)

        if aug:
            kernel = th.rand((patches.shape[0], 1, 1, 1)).to(self.device)
            patches = patches + aug_noise * kernel * th.randn_like(patches, requires_grad=False)
        return patches
