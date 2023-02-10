
# ====================================================
#  deepstyle  -  a neural style project
#  neural networks and deep learning exam project
#
#   UNIPD Project |  AY 2022/23  |  NNDL
#   group : Barone, Ninni, Zinesi
# ----------------------------------------------------
#   coder : Barone Francesco
#         :   github.com/baronefr/
#   dated : 5 Feb 2023
#     ver : 1.0.0
# ====================================================


from deepstyle.image import ImageManager



def compute_feat( model, image, layers, fn = lambda x : x, device = None) -> list:
    if isinstance(image, ImageManager):
        if device is None:  device = image.device
        return [ fn(x).detach().to(device) for x in model(image.data, layers) ]
    else:
        return [ fn(x).detach().to(device) for x in model(image, layers) ]
