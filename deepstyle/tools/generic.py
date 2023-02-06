
from deepstyle.image import ImageManager



def compute_feat( model, image, layers, fn = lambda x : x, device = None) -> list:
    if isinstance(image, ImageManager):
        if device is None:  device = image.device
        return [ fn(x).detach().to(device) for x in model(image.data, layers) ]
    else:
        return [ fn(x).detach().to(device) for x in model(image, layers) ]
