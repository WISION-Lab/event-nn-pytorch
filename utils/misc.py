import torch


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach().numpy()
    else:
        return x


def get_pytorch_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def rotate_points(points, theta, center_x=0.0, center_y=0.0):
    translate = torch.tensor(
        [
            [1.0, 0.0, -center_x],
            [0.0, 1.0, -center_y],
            [0.0, 0.0, 1.0],
        ]
    )
    theta = torch.deg2rad(theta)
    rotate = torch.tensor(
        [
            [torch.cos(theta), torch.sin(theta), 0.0],
            [-torch.sin(theta), torch.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Convert to homogeneous coordinates.
    points = torch.concat([points, torch.ones_like(points[..., :1])], dim=-1)

    # Right multiply transposed transforms because of the shape of
    # points (the last dimension is spatial).
    points = points @ translate.t() @ rotate.t() @ torch.inverse(translate).t()

    # Convert back from homogeneous coordinates
    return points[..., :-1]


def torch_rand_uniform(low, high, size=()):
    return low + (high - low) * torch.rand(size)
