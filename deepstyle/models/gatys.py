
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


import torch.nn as nn
from torchvision import transforms
import numpy as np


# these values are used to preprocess
# the input of pytorch VGG19
vgg19_mean = np.array( [0.485, 0.456, 0.406] )
vgg19_std  = np.array( [0.229, 0.224, 0.225] )



class gatys( nn.Module ):
    """Network for Neural Style Transfer, built on top of a pre-trained VGG19 to compute features extracted from intermediate layers."""

    def __init__(self, input_model, forward_hooks:list,
                 pool:str = 'max',  crop_model:bool = True
                ):
        super().__init__()

        # -----------------------------
        #       HOOKS MANAGEMENT
        # -----------------------------
        self.hooks, self.hook_handlers = {}, []
        def takeOutput(name:str):
            def hook(model, input, output):
                self.hooks[name] = output #.detach()
                # note: DO NOT DETACH! I WILL NEED GRADIENTS LATER
            return hook
        #
        #  self.hook_handlers     will store the handlers of
        #                         fwd hooks I will register
        #  self.hooks             will store the values taken with hooks
        #                         from selected layers

        self.hooks_init = forward_hooks.copy()  # keep a backup copy of this arg
        todo_hooks = set(forward_hooks)         # will use this list while I loop on network layers


        # --------------------------------------
        #              MODEL BUILD
        # --------------------------------------
        #   I loop over the input model layers,
        #  registering a hook when requested.
        # --------------------------------------
        self.model = nn.Sequential()
        
        conv_num, conv_group = 1, 1
        for layer in input_model.children():
            
            if isinstance(layer, nn.Conv2d):
                name = 'conv{}_{}'.format(conv_group, conv_num)
            
            elif isinstance(layer, nn.ReLU):
                name = 'relu{}_{}'.format(conv_group, conv_num)
                #  ^^  takes the name similar to last conv layer
                conv_num += 1

            elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d) ):
                # take name & reset counters
                name = 'pool_{}'.format(conv_group)
                conv_group += 1;   conv_num = 1; 

                # replace pool layer with desired choice
                if   pool.lower() == 'avg':
                    layer = nn.AvgPool2d( kernel_size = layer.kernel_size,
                                          stride = layer.stride )
                elif pool.lower() == 'max':
                    layer = nn.MaxPool2d( kernel_size = layer.kernel_size,
                                          stride = layer.stride )
                else:
                    raise RuntimeError('not valid pool arg ({})'.format(pool))

            else: name = 'buh'
            
            self.model.add_module(name, layer)

            if name in todo_hooks:
                # register a hook for current layer and track its handler
                self.hook_handlers.append(
                    self.model[-1].register_forward_hook( takeOutput(name) )
                )
                todo_hooks.remove(name)

            if crop_model:
                if not todo_hooks: break
        
        # fix the weights, to be sure
        for param in self.model.parameters():
            param.requires_grad = False

        # set to eval mode
        self.model.eval()


    def forward(self, x, query : list = None):

        self.hooks = {}     # reset hooks dict

        _ = self.model(x)   # I don't care about this ...

        # ... returns the features extracted by hooks
        if query is not None:
            return [ self.hooks[key] for key in query ]
            # this is a list
        else:
            return self.hooks
            # this is a dict


    # preprocess an input to VGG19
    make_input = transforms.Compose([
        transforms.Normalize(mean = vgg19_mean, std=vgg19_std),
        transforms.Lambda(lambda x: x.mul_(255)),
    ])

    # revert the preprocess of VGG19
    make_output = transforms.Compose([
        transforms.Lambda(lambda x: x.mul_(1./255)),
        transforms.Normalize(mean = [ 0., 0., 0. ], std = 1/vgg19_std),
        transforms.Normalize(mean = -vgg19_mean, std = [1,1,1]),
    ])