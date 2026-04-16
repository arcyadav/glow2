import numpy as np
from PIL import Image
import threading


def save_image(x, path):
    im = Image.fromarray(x)
    im.save(path, optimize=True)


# Assumes input is torch tensor OR numpy in [NCHW]
def save_raster(x, path, rescale=False, width=None):
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()

    t = threading.Thread(target=_save_raster, args=(x, path, rescale, width))
    t.start()


def _save_raster(x, path, rescale, width):
    x = to_raster(x, rescale, width)
    save_image(x, path)


# Shape: (N, H, W, C)
def to_raster(x, rescale=False, width=None):
    # Convert NCHW → NHWC if needed
    if x.shape[1] in [1, 3]:  # assume NCHW
        x = np.transpose(x, (0, 2, 3, 1))

    if len(x.shape) == 3:
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))

    if x.shape[3] == 1:
        x = np.repeat(x, 3, axis=3)

    if rescale:
        x = (x - x.min()) / (x.max() - x.min() + 1e-8) * 255.

    x = np.clip(x, 0, 255).astype(np.uint8)

    n_batch = x.shape[0]

    if width is None:
        width = int(np.ceil(np.sqrt(n_batch)))

    height = int(np.ceil(n_batch / width))

    tile_h, tile_w = x.shape[1], x.shape[2]

    result = np.zeros((height * tile_h, width * tile_w, 3), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            idx = i * width + j
            if idx >= n_batch:
                break
            result[
                i * tile_h:(i + 1) * tile_h,
                j * tile_w:(j + 1) * tile_w
            ] = x[idx]

    return result