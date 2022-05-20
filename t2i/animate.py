import glm, gc, torch as th, math, os, sys
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image, ImageFilter
import numpy as np


def video(frames, output_file, fps=14, **kwargs):
    '''
    '''
    import imageio, tqdm

    writer = imageio.get_writer(output_file, fps=fps)
    for im in tqdm.tqdm(frames):
        writer.append_data(imageio.imread(im))
    writer.close()


infer_helper = None
def AdaBins(path):
    global infer_helper
    cwd = os.getcwd()
    if infer_helper is None:
        try:
            sys.path.append(path)
            os.chdir(path)
            from infer import InferenceHelper
            infer_helper = InferenceHelper(dataset='nyu')
        except Exception as ex:
            print(ex)
            sys.exit(1)
        finally:
            os.chdir(cwd)
    return infer_helper

def adabins_depth(pil_image, adabins_dir=None, max_depth_area=500_000):

    infer_helper = AdaBins(adabins_dir if adabins_dir else os.environ['ADABINS_DIR'])

    # `max_depth_area` -> if the area of an image is above this, the depth model fails
    width, height = pil_image.size
    image_area = width * height
    if image_area > max_depth_area:
        depth_scale = math.sqrt(max_depth_area / image_area)
        height, width = int(height * depth_scale), int(width * depth_scale)
        depth_input = pil_image.resize((width, height), Image.LANCZOS)
        depth_resized = True
    else:
        depth_input = pil_image
        depth_resized = False

    gc.collect()
    th.cuda.empty_cache()

    _, depth_map = infer_helper.predict_pil(depth_input)

    gc.collect()
    th.cuda.empty_cache()

    return depth_map, depth_resized


@th.no_grad()
def apply_grid(tensor, grid, border_mode, sampling_mode):
    height, width = tensor.shape[-2:]
    if border_mode == 'wrap':
        max_offset = th.max(grid.clamp(min= 1))
        min_offset = th.min(grid.clamp(max=-1))
        max_coord = max(max_offset, abs(min_offset))
        if max_coord > 1:
            mod_offset = int(math.ceil(max_coord))
            mod_offset += 1 - (mod_offset % 2)
            grid = grid.add(mod_offset).remainder(2.0001).sub(1)
    return F.grid_sample(tensor, grid, mode=sampling_mode, align_corners=True, padding_mode=border_mode)


@th.no_grad()
def apply_flow(img, flow, border_mode='mirror', sampling_mode='bilinear_plane', device='cuda'):
    try:
        tensor = img.get_image_tensor().unsqueeze(0)
        fallback = False
    except NotImplementedError:
        tensor = TF.to_tensor(img.decode_image()).unsqueeze(0)
        tensor.to(device)
        fallback = True

    height, width = flow.shape[-2:]
    identity = th.eye(3).to(device)
    identity = identity[0:2,:].unsqueeze(0) #for batch
    uv = F.affine_grid(identity, tensor.shape, align_corners=True)
    size = th.tensor([[[[width/2,height/2]]]], device=device)
    flow = TF.resize(flow, tensor.shape[-2:]).movedim(1,3).div(size)
    grid = uv - flow
    tensor = apply_grid(tensor, grid, border_mode, sampling_mode)
    if not fallback:
        img.set_image_tensor(tensor.squeeze(0))
        tensor_out = img.decode_tensor().detach()
    else:
        array = tensor.squeeze().movedim(0,-1).mul(255).clamp(0, 255)
        array = array.cpu().detach().numpy().astype(np.uint8)[:,:,:]
        img.encode_image(Image.fromarray(array))
        tensor_out = tensor.detach()
    return tensor_out


@th.no_grad()
def render_image_3d(image, depth, P, T, border_mode, sampling_mode, stabilize, device='cuda'):
    h, w = image.shape[-2:]
    f = w / h
    image = image.unsqueeze(0)

    y,x = th.meshgrid(th.linspace(-1,1,h),th.linspace(-f,f,w))
    x = x.unsqueeze(0).unsqueeze(0)
    y = y.unsqueeze(0).unsqueeze(0)
    xy = th.cat([x,y], dim=1).to(device)

    identity = th.eye(3).to(device)
    identity = identity[0:2,:].unsqueeze(0)
    uv = F.affine_grid(identity, image.shape, align_corners=True)
    # get the depth at each point
    depth = depth.unsqueeze(0).unsqueeze(0)

    view_pos = th.cat([xy,-depth,th.ones_like(depth)],dim=1)
    # apply the camera move matrix
    next_view_pos = th.tensordot(T.float(), view_pos.float(), ([0],[1])).movedim(0,1)

    # apply the perspective matrix
    clip_pos = th.tensordot(P.float(), view_pos.float(), ([0],[1])).movedim(0,1)
    clip_pos = clip_pos/(clip_pos[:,3,...].unsqueeze(1))

    next_clip_pos = th.tensordot(P.float(), next_view_pos.float(), ([0],[1])).movedim(0,1)
    next_clip_pos = next_clip_pos/(next_clip_pos[:,3,...].unsqueeze(1))

    # get the offset
    offset = (next_clip_pos - clip_pos)[:,0:2,...]

    # render the image
    if stabilize:
        advection = offset.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
        offset = offset - advection

    offset = offset.permute(0,2,3,1)
    grid = uv - offset

    return apply_grid(image, grid, border_mode, sampling_mode).squeeze(0), offset.squeeze(0)

#-------------------------------------------------------------------------------
math_env, global_t, eval_memo = None, 0, {}
def parametric_eval(string, **vals):
    global math_env
    if string in eval_memo: return eval_memo[string]

    if isinstance(string, str):
        if math_env is None:
          math_env = dict(abs=abs, max=max, min=min, pow=pow, round=round, __builtins=None)
          math_env.update({key: getattr(math, key) for key in dir(math) if '_' not in key})
        math_env.update(vals)
        math_env['t'] = global_t
        try: output = eval(string, math_env)
        except SyntaxError as e:
            raise RuntimeError(f'error <!> :: invalid eval `{string}`.')
        eval_memo[string] = output
        return output
    else:
        return string

#-------------------------------------------------------------------------------
def set_t(t):
  global global_t, eval_memo
  global_t = t
  eval_memo = {}


@th.no_grad()
def zoom_3d(pil_image, t, translate=(0,0,0), rotate=0, field_of_view=45, near_plane=180, far_plane=15000,
            border_mode='mirror', sampling_mode='bilinear_plane', stabilize=False,
            device='cuda', verbose=False, **kwargs):
    set_t(t)

    width, height = pil_image.size
    px = 2 / height
    f = width / height

    alpha = math.radians(field_of_view)
    depth = 1 / (math.tan(alpha / 2))
    #---------------------------------------------------------------------------
    # convert depth map
    depth_map, depth_resized = adabins_depth(pil_image)
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    #---------------------------------------------------------------------------
    depth_map = np.interp(depth_map, (1e-3, 10), (near_plane * px, far_plane * px))

    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    #---------------------------------------------------------------------------
    depth_median = np.median(depth_map.flatten())
    depth_mean   = np.mean(depth_map)
    r = depth_min / px
    R = depth_max / px
    mu = (depth_mean+depth_median)/(2*px)
    if verbose: print(f'depth range: {r} to {R}\nmu: {mu}')
    #---------------------------------------------------------------------------
    translate = [parametric_eval(x, r=r, R=R, mu=mu) for x in translate]
    rotate = parametric_eval(rotate, r=r, R=R, mu=mu)
    tx,ty,tz = translate
    if verbose: print(f't={t}\ntranslation: {translate}\nrotation: {rotate}')
    #---------------------------------------------------------------------------
    interpolation = TF.InterpolationMode.BICUBIC
    try:
        image_tensor = T.ToTensor()(pil_image).to(device)
        depth_map = th.from_numpy(depth_map)
        depth_tensor = TF.resize(depth_map, image_tensor.shape[-2:], interpolation)
        depth_tensor = depth_tensor.squeeze().to(device)
        fallback = False
    except NotImplementedError:
        image_tensor = TF.to_tensor(pil_image).to(device)
        if depth_resized:
            depth_tensor = th.from_numpy(depth_map)
            depth_tensor = TF.resize(depth_tensor, image_tensor.shape[-2:], interpolation)
            depth_tneosr = depth_tensor.squeeze().to(device)
        else:
            depth_tensor = th.from_numpy(depth_map).squeeze().to(device)
        fallback = True
    #---------------------------------------------------------------------------
    p_matrix = th.as_tensor(glm.perspective(alpha, f, 0.1, 4).to_list()).to(device)
    r_matrix = glm.mat4_cast(glm.quat(*rotate))
    t_matrix = glm.translate(glm.mat4(1), glm.vec3(tx * px, -ty * px, tz * px))
    #---------------------------------------------------------------------------
    T_matrix = th.as_tensor((r_matrix @ t_matrix).to_list()).to(device)
    new_image, flow = render_image_3d(image_tensor, depth_tensor, p_matrix, T_matrix,
    border_mode=border_mode, sampling_mode=sampling_mode, stabilize=stabilize)
    #---------------------------------------------------------------------------
    flow = flow.div(2).mul(th.tensor([[[[width,height]]]], device=device))
    #---------------------------------------------------------------------------
    if not fallback:
        image = T.ToPILImage()(new_image)
    else:
        tensor = new_image.movedim(0,-1).mul(255).clamp(0, 255)
        array = tensor.cpu().detach().numpy().astype(np.uint8)[:,:,:]
        image = Image.fromarray(array)
    return image
