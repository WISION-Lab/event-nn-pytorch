import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as func


def as_float32(x):
    if isinstance(x, torch.Tensor) and x.dtype == torch.uint8:
        return x.float() / 255.0
    elif isinstance(x, np.ndarray) and x.dtype == np.uint8:
        return x.astype(np.float32) / 255.0
    elif type(x) in (tuple, list) and isinstance(x[0], int):
        return type(x)(x_i / 255.0 for x_i in x)
    else:
        return x


def as_uint8(x):
    if isinstance(x, torch.Tensor) and x.dtype != torch.uint8:
        # noinspection PyUnresolvedReferences
        return (x * 255.0).byte()
    elif isinstance(x, np.ndarray) and x.dtype != np.uint8:
        return (x * 255.0).astype(np.uint8)
    elif type(x) in (tuple, list) and isinstance(x[0], float):
        return type(x)(int(x_i * 255.0) for x_i in x)
    else:
        return x


def centered_pad_and_crop(x, size, fill=0, padding_mode="constant"):
    total_x = size[1] - x.shape[-1]
    pad_l = total_x // 2
    pad_r = total_x - pad_l
    total_y = size[0] - x.shape[-2]
    pad_t = total_y // 2
    pad_b = total_y - pad_t
    return func.pad(
        x, [pad_l, pad_t, pad_r, pad_b], fill=fill, padding_mode=padding_mode
    )


def pad_to_multiple(x, multiple, fill=0, padding_mode="constant"):
    pad_w = -x.shape[-1] % multiple
    pad_h = -x.shape[-2] % multiple
    return func.pad(x, [0, 0, pad_w, pad_h], fill=fill, padding_mode=padding_mode)


def read_image(filename):
    return torchvision.io.read_image(str(filename))


def read_video(filename):
    return torchvision.io.read_video(str(filename))


def write_image(filename, image, **kwargs):
    filename = str(filename)
    lower = filename.lower()
    image = torch.as_tensor(image)
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        torchvision.io.write_jpeg(image, filename, **kwargs)
    elif lower.endswith(".png"):
        torchvision.io.write_png(image, filename, **kwargs)
    else:
        raise ValueError("The image extension must be .jpg, .jpeg, or .png.")


def write_video(filename, video, fps=30, is_chw=True):
    filename = str(filename)
    video = torch.as_tensor(video)
    if is_chw:
        video = video.permute(0, 2, 3, 1)
    torchvision.io.write_video(filename, video, fps=fps)
