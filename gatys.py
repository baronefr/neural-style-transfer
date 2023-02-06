#!/usr/bin/python3

# ========================================
#  deepstyle  -  a neural style project
# ----------------------------------------
#   coder : Barone Francesco
#         :   github.com/baronefr/
#   dated : 5 Feb 2023
#     ver : 0.1.1
# ========================================




# %% import libraries

import torch
import torchvision

import deepstyle
from deepstyle.tools.loss import gramfn, GramLoss
from deepstyle.tools.generic import compute_feat
from deepstyle.tools.loss import LossManager
from deepstyle.image import ImageManager

import argparse
parser = argparse.ArgumentParser()


# %% setup env

torch.manual_seed(17)

# remark: device is parsed by command line ...
#device = "cuda" if torch.cuda.is_available() else "cpu"
#device



# %% parse arguments

parser.add_argument("-style", help="style image", type=str, default = "./image/starry_night.jpg")
parser.add_argument("-content", help="content image", type=str, default = "./image/padova-prato.jpg")
parser.add_argument("-output", help="output image destination", type=str, default = None)
parser.add_argument("-image-size", help="resize parameter of input images", type=int, default=512)
parser.add_argument("-color-control",  help="color control mode", type=str, default='none')

parser.add_argument("-iterations", help="number of iterations for NST algo", type=int, default=400)
parser.add_argument("-optimizer", help="optimizer", type=str, default='LBFGS')
parser.add_argument("-learn-rate", help="learning rate of optimizer", type=float, default=1e0)

parser.add_argument("-content-layers", help="layers for content (comma separated)", type=str, default='conv4_2')
parser.add_argument("-style-layers", help="layers for style (comma separated)", type=str, default='conv1_1,conv2_1,conv3_1,conv4_1,conv5_1')
# ^^^ these default values are taken from Gatys paper
parser.add_argument("-content-weight", help="content loss coefficient", type=float, default=1e0)
parser.add_argument("-content-weight-hr", help="content loss coefficient at high-res mode", type=float, default=None)
parser.add_argument("-style-weights", help="style loss weights (one per layer)", type=str, default=[ 1e2/n**2 for n in [64,128,256,512,512] ])

parser.add_argument("-high-res", help="resolution of image upscaling", type=int, default=-1)
#           remark: high-res = 820 requires 7.7GB of GPU memory                  ^^^^^^^^^^
parser.add_argument("-iterations-high-res", help="number of iterations for NST algo at higher resolution", type=int, default=200)

parser.add_argument("-device", help="learning rate of optimizer", type=str, default=None)

config = parser.parse_args()
print('this config:', config )

content_layers = config.content_layers.split(",")
style_layers = config.style_layers.split(",")

IMAGE_SIZE = config.image_size
COLOR_CONTROL = config.color_control
HIGH_RES = config.high_res  # set to any < IMAGE_SIZE value to disable

STYLE_IMAGE_FILE   = config.style
CONTENT_IMAGE_FILE = config.content
OUTPUT_IMAGE = config.output

TRANSFER_STEPS = config.iterations
TRANSFER_STEPS_HR = config.iterations_high_res

OPT = config.optimizer
OPT_LR = config.learn_rate

OPT_CONTENT_WEIGHT = config.content_weight
OPT_CONTENT_WEIGHT_HR = config.content_weight
if OPT_CONTENT_WEIGHT_HR is None:
    OPT_CONTENT_WEIGHT_HR = OPT_CONTENT_WEIGHT

OPT_STYLE_WEIGHTS = config.style_weights
if not isinstance(OPT_STYLE_WEIGHTS, list):
    OPT_STYLE_WEIGHTS = str(OPT_STYLE_WEIGHTS)
if OPT_STYLE_WEIGHTS in ['none', 'None']:
    OPT_STYLE_WEIGHTS = None
elif isinstance(OPT_STYLE_WEIGHTS, str):
    OPT_STYLE_WEIGHTS = [ float(x) for x in OPT_STYLE_WEIGHTS.split(',') ]

device = config.device
if device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    if ',' in device:
        device, device_hr = device.split(',')
    else:
        device_hr = device

LOGGER_PERIOD = 20


# %% define the NN to use

# take the original nst model (pretrained weights)
vgg19 = torchvision.models.vgg19(weights='IMAGENET1K_V1')

nstn = deepstyle.models.gatys(vgg19.features, content_layers + style_layers)
nstn.to(device) 
# requires 700 MB of GPU memory

# %% get the images

style_img = ImageManager(STYLE_IMAGE_FILE, resize = IMAGE_SIZE, device=device,
                make_input = nstn.make_input, make_output = nstn.make_output )
content_img = ImageManager(CONTENT_IMAGE_FILE, resize = IMAGE_SIZE, device=device,
                make_input = nstn.make_input, make_output = nstn.make_output )

# note: the image is not actually processed until you call load
#style_img.load()
#content_img.load()

# to have a preview of style image
#style_img.tens2pil()



# %% process the images
# requires few MB of GPU memory

cc = deepstyle.image.color_control(COLOR_CONTROL, [style_img], [content_img])
target = cc.preprocess()
generated_img = target[0]




# %%
# requires 700 MB of GPU memory

target_style_feat = compute_feat(nstn, style_img, style_layers, gramfn)
target_content_feat = compute_feat(nstn, content_img, content_layers)



# %% set up the loss trackers & train process

catStyle   = LossManager(style_layers, target_style_feat, GramLoss(), weights = OPT_STYLE_WEIGHTS )
# ^ weights = [ 1e2/n**2 for n in [64,128,256,512,512] ]
# v scale = 1e0
catContent = LossManager(content_layers, target_content_feat, torch.nn.MSELoss(), scale = OPT_CONTENT_WEIGHT )  #reduction='sum'

if OPT == 'LBFGS':
    optimizer = torch.optim.LBFGS([generated_img.data], lr=OPT_LR)
elif OPT == 'Adam':
    optimizer = torch.optim.Adam([generated_img.data], lr=OPT_LR)
else:
    raise Exception('unknown optimizer')

transfer = deepstyle.tools.train.TrainMethod(optimizer = optimizer, logger_period = LOGGER_PERIOD)



def closure() -> torch.Tensor:
    """Function that performs training"""
    global transfer

    transfer.optimizer.zero_grad()
    
    feats = nstn( generated_img.data )

    loss_style = catStyle.compute_loss_from_dict( feats )
    loss_cont = catContent.compute_loss_from_dict( feats )
    loss = loss_style + loss_cont

    loss.backward()

    transfer.logger_update( msg = "loss = {}".format( loss.item() ) )
    transfer.iter_count += 1
    return loss

transfer.set_closure( closure )


# %% NST loop ===============================================
# requires 1700 MB of GPU memory

transfer.loop(iter_max = TRANSFER_STEPS, desc = 'Neural Style Transfer')


# %% show result

res = cc.postprocess( generated_img )
res


# %% high resolution preprocessing =================

if HIGH_RES > IMAGE_SIZE:
    # basically, I will run again the same algo ...

    # account for device change
    if device != device_hr:
        device_hr
        print('[info] switching device {} -> {}'.format(device, device_hr) )
        nstn.to( device_hr )
        style_img.device = device_hr
        content_img.device = device_hr
        generated_img.device = device_hr

    # ... using higher-resolution style & content
    #style_img = ImageManager(STYLE_IMAGE_FILE, resize = HIGH_RES, device=device_hr,
    #            make_input = nstn.make_input, make_output = nstn.make_output )
    #content_img = ImageManager(CONTENT_IMAGE_FILE, resize = HIGH_RES, device=device_hr,
    #            make_input = nstn.make_input, make_output = nstn.make_output )
    style_img.size = HIGH_RES
    content_img.size = HIGH_RES

    cc = deepstyle.image.color_control(COLOR_CONTROL, [style_img], [content_img])
    cc.preprocess(force_target = False)

    # ... and initializing the algo from the generated output at lower resolution
    generated_img.resize( content_img.data.shape[2:] )
    if device != device_hr: # force to switch device
        generated_img.data = generated_img.data.to(device)
    generated_img.data.requires_grad = True


    # from now on it is the same as before
    target_style_feat = compute_feat(nstn, style_img, style_layers, gramfn)
    target_content_feat = compute_feat(nstn, content_img, content_layers)

    catStyle   = LossManager(style_layers, target_style_feat, GramLoss(), weights = OPT_STYLE_WEIGHTS )
    # ^ weights = [ 1e2/n**2 for n in [64,128,256,512,512] ]
    # v scale = 1e0
    catContent = LossManager(content_layers, target_content_feat, torch.nn.MSELoss(), scale = OPT_CONTENT_WEIGHT_HR )  #reduction='sum'

    if OPT == 'LBFGS':
        optimizer = torch.optim.LBFGS([generated_img.data], lr=OPT_LR)
    elif OPT == 'Adam':
        optimizer = torch.optim.Adam([generated_img.data], lr=OPT_LR)
        

    transfer = deepstyle.tools.train.TrainMethod(optimizer = optimizer, logger_period = LOGGER_PERIOD)


    def closure() -> torch.Tensor:
        """Function that performs training"""
        global transfer

        transfer.optimizer.zero_grad()
        
        feats = nstn( generated_img.data )

        loss_style = catStyle.compute_loss_from_dict( feats )
        loss_cont = catContent.compute_loss_from_dict( feats )
        loss = loss_style + loss_cont

        loss.backward()

        transfer.logger_update( msg = "loss = {}".format( loss.item() ) )
        transfer.iter_count += 1
        return loss


    transfer.set_closure( closure )

    
# %% train loop for high-res ===================================

if HIGH_RES > IMAGE_SIZE:
    transfer.loop(iter_max = TRANSFER_STEPS_HR, desc = 'high-res Style Transfer')
    
# %% show the result of high-res

if HIGH_RES > IMAGE_SIZE:
    res = cc.postprocess( generated_img )
    res

# %%

if OUTPUT_IMAGE is not None:
    # remark: this is a PIL image, so you can save it the usual way
    cc.postprocess( generated_img ).save( OUTPUT_IMAGE )

print('end of script')

