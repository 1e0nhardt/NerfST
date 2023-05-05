from PIL import Image
import numpy as np
import torch
import itertools

def prepare_style_image(image_filename, scale_factor=1.0, alpha_color=None):
    pil_image = Image.open(image_filename)

    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)

    image = np.array(pil_image, dtype="uint8")

    if len(image.shape) == 2:
        image = image[:, :, None].repeat(3, axis=2)

    assert len(image.shape) == 3
    assert image.dtype == np.uint8
    assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."

    image = torch.from_numpy(image.astype("float32") / 255.0)

    if alpha_color is not None and image.shape[-1] == 4:
        assert image.shape[-1] == 4
        image = image[:, :, :3] * image[:, :, -1:] + alpha_color * (1.0 - image[:, :, -1:])
    else:
        image = image[:, :, :3]
        
    return image


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())