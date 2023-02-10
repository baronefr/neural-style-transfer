
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


import numpy as np
import PIL
from PIL import Image

import torch
from torchvision import transforms



toTen = transforms.Compose([ transforms.ToTensor() ])
toPIL = transforms.Compose([ transforms.ToPILImage() ])



class color_util():

    def hist_match(S : np.ndarray, C : np.ndarray, variant = 'eigs'):
        """Linear transformation of S such that mean and covariance of the RGB values
           in the new style image S' match those of C"""

        # expected input | ndarray, dtype float64, in range [0,1],
        #                | with shape (H, W, ch = 3)
        assert isinstance( S,  np.ndarray )
        assert isinstance( C,  np.ndarray )
        assert variant in ['cholesky', 'eigs']

        # I am forced to correct this in the library...
        if(C.shape[2] != 3):
            C = C.transpose(1,2,0)

        # 0) check that arrays are made of elements in range [0,1]
        #    and try to force this condition
        if( np.mean(S) > 10 ):   S = S/255
        if( np.mean(C) > 10 ):   C = C/255


        # 1) compute mean and covariance of pixel values
        muS = np.mean( S, axis=(0,1) )
        S_linear = (S - muS).transpose(2,0,1).reshape(3,-1)
        covS = np.dot( S_linear, S_linear.T )/S_linear.shape[1]

        muC = np.mean( C, axis=(0,1) )
        C_linear = (C - muC).transpose(2,0,1).reshape(3,-1)
        covC = np.dot( C_linear, C_linear.T )/C_linear.shape[1]

        # add a small value to covariance matrix to avoid singularity when computing inverse matrix
        covS += np.eye(covS.shape[0])*1e-8
        covC += np.eye(covC.shape[0])*1e-8


        # 2) get the transform matrix 
        if variant == 'cholesky':
            L_S = np.linalg.cholesky(covS)
            L_C = np.linalg.cholesky(covC)
            A = L_C @ np.linalg.inv(L_S)
            # A is a 3x3 matrix now

        elif variant == 'eigs':
            # eigenvectors, eigenvalues  <- linalg.eigh( M )
            lambdaS, uS = np.linalg.eigh(covS)
            sigmaS = uS @ ( np.sqrt(np.diag(lambdaS)) ) @ uS.T

            lambdaC, uC = np.linalg.eigh(covC)
            sigmaC = uC @ ( np.sqrt(np.diag(lambdaC)) ) @ uC.T

            A = sigmaC @ np.linalg.inv(sigmaS)


        # 3) apply the chosen linear transform
        b = muC  #- ( A @ muS )  # IDK, but it makes the image darker...
        newS = (A @ S_linear).transpose(1,0) + b
        # I transpose A@S_l to broadcast along channel dim

        # remove burned pixels
        newS[ newS > 1 ] = 1
        newS[ newS < 0 ] = 0

        # - the image is still linearized ... reshape!
        # - PIL likes it as an array of int values...
        return (newS*225).astype(np.uint8).reshape( S.shape[0], S.shape[1], 3 )
    
    def luminance_forward(img):
        """This function computes YYY repres. of a given image"""
        
        # note:  YUV works as well!
        gg = img.convert('YCbCr').getchannel('Y')
        return Image.merge('RGB', (gg, gg, gg) )

    def luminance_revert( src, target, revert_fn = lambda x : x):
        """returns target image taking the luminance channel from src image."""
        # takes luminance from generated image ...
        gg = revert_fn(src).convert('YCbCr').getchannel('Y')

        # ... and color from the target image
        cc = revert_fn(target).convert('YCbCr')

        merged = Image.merge('YCbCr', (gg, cc.getchannel('Cb'), cc.getchannel('Cr')) )
        return merged.convert('RGB')







class color_control():

    def __init__(self, mode, style : list, content : list):
        
        assert mode in ['none', 'luminance', 'luminance_full', 'hist', 'hist_from_style' ], 'not valid option'
        assert isinstance(style, list)  , 'must provide a list of images'
        assert isinstance(content, list), 'must provide a list of images'

        self.mode = mode
        self.style = style
        self.content = content
        self.target = None
        self.keep = None


    def preprocess(self, loadargs : dict = {}, init_from : str = 'content',
                   force_target : list = None, add_noise : float | None = None,
                   hist_association : list = None, hist_variant : str = 'eigs') -> dict:
        
        assert init_from in ['content', 'style', 'rand'], 'init source not valid'

        if self.mode in ['none', 'luminance']:
            # default processing
            for img in self.style:   img.load(**loadargs)
            for img in self.content: img.load(**loadargs)
            # ^  this is kept to restore colors later


        elif self.mode == 'luminance_full':
            # use the luminance channel
            for img in self.style:   img.load(preprocess = color_util.luminance_forward, **loadargs)
            for img in self.content: img.load(preprocess = color_util.luminance_forward, **loadargs)
            # ^  this is kept to restore colors later


        elif self.mode in ['hist']:
            # transfer color from content to style (with hist matching)
            for img in self.content: img.load(**loadargs)

            # load the content as image
            content_tmp = [ img.original_data for img in self.content ]

            class wrap_hist_match():
                """wrapper for target fixed hist_match"""
                def __init__(self, target):
                    self.target = target
                def call(self, obj):
                    return Image.fromarray(
                        color_util.hist_match( np.array(obj), np.array(self.target), variant = hist_variant )
                    )

            if hist_association is None:
                hist_association = [0]*len(self.style)

            for img, idx in zip(self.style, hist_association):
                wr = wrap_hist_match(content_tmp[idx])
                img.load(preprocess = wr.call, **loadargs)

            del content_tmp


        elif self.mode in ['hist_from_style']:
            # transfer color from style to content (with hist matching)
            for img in self.style:   img.load(**loadargs)

            style_tmp = [ img.original_data for img in self.style ]

            class wrap_hist_match():
                """wrapper for target fixed hist_match"""
                def __init__(self, target):
                    self.target = target
                def call(self, obj):
                    return Image.fromarray(
                        color_util.hist_match( np.array(obj), np.array(self.target), variant=hist_variant )
                    )
            
            if hist_association is None:
                hist_association = [0]*len(self.content)

            for img, idx in zip(self.content, hist_association):
                wr = wrap_hist_match(style_tmp[idx])
                img.load(preprocess = wr.call, **loadargs)

            del style_tmp


        
        
        if force_target is None:

            self.target = []

            if init_from == 'content':
                # copy the target from content image
                for img in self.content:
                    tgt = img.soft_clone()
                    tgt.data = torch.autograd.Variable(img.data.clone())#, requires_grad=True) # .to(img.device)
                    if add_noise is not None:
                        tgt.data += add_noise*torch.rand(tgt.data.shape, device = tgt.data.device)
                    tgt.data.requires_grad = True
                    self.target.append( tgt )

            elif init_from == 'style':
                for img, etc in zip(self.style, self.content):
                    tgt = img.soft_clone()
                    # nb: tgt size will be broken!
                    tgt.data = torch.autograd.Variable( 
                        # take new image size from content picture
                        transforms.Resize( etc.data.shape[2:] )( img.data ).clone(),
                        requires_grad=True
                    )
                    self.target.append( tgt )

            else:
                # TODO: use https://pytorch.org/docs/stable/generated/torch.rand.html
                raise Exception('random init feat not yet implemented')
        
        elif force_target is False:
            return None

        else:
            self.target = force_target

        return self.target



    def postprocess(self, generated_img : list, content_idx = 0):
        if self.mode in ['luminance', 'luminance_full']:
            print('warning: overwriting content image without preprocessing')
            self.content[content_idx].load()
            # cc with luminance requires a revert of image coloring
            res = color_util.luminance_revert( generated_img.tens2pil(), self.content[content_idx].tens2pil() )
        else:
            res = generated_img.tens2pil()

        return res













class ImageManager():

    def __init__(self, origin, resize : int = 512, device = None, make_input = None, make_output = None):
        
        assert isinstance(origin, (str, np.ndarray, PIL.Image.Image) ), \
            'origin input must be a string, ndarray or PIL image'

        self.origin = origin
        self.size = resize
        self.make_input = make_input
        self.make_output = make_output
        self.device = device
        self.data = None # data will be allocated later


    def __str__(self):
        return "custom image object with origin" + self.origin

    def soft_clone(self):
        return ImageManager(origin = self.origin, resize = self.size, 
                            device = self.device,
                            make_input = self.make_input, 
                            make_output = self.make_output
                )

    def volatile_load_PIL(self):
        return Image.open(self.input)

    def todevice(self):
        self.data.to(self.device)

    def load(self, preprocess = None, make = 'default', resize : bool = True,
             device : bool = True, add_batch : bool = True, rewrite : bool = True) -> torch.Tensor:

        if isinstance(self.origin, str):
            # load the image
            img = Image.open(self.origin)
        else:
            # assert that this is a PIL image or ndarray...
            img = self.origin
        
        self.original_data = img

        if preprocess:  img = preprocess(img)
        
        img = toTen( img )  # from now on this is a tensor

        if device:     img = img.to(self.device)

        if resize:     img = transforms.Resize(self.size)( img )

        # optional: apply custom preprocess function, else use the default
        if make == 'default':
            img = self.make_input(img)
        elif make not in ['none', None]:
            img = make(img)

        # optional: add dummy batch dimention
        if add_batch:  img = img.unsqueeze(0)

        if rewrite:    self.data = img
        return img
    

    def make_tensor(self):
        self.data = torch.Tensor( self.origin )
    def make_PIL(self):
        self.data = PIL.Image( self.origin )
    def make_nparray(self):
        self.data = np.array( self.origin )


    def tens2pil(self, revert = 'default', clamp : bool = True):

        assert isinstance(self.data, torch.Tensor ), \
            'data must be a pytorch tensor to call this function'

        x = self.data.clone().detach()

        # invert the input trasform
        if revert == 'default':
            x = self.make_output(x)
        elif revert not in ['none', None]:
            x = revert(x)

        # remove burned areas
        if clamp:   x = torch.clamp( x, min=0, max=1)

        return toPIL( torch.squeeze(x) )
    
    def clone(self):
        assert isinstance(self.data, torch.Tensor ), \
            'data must be a pytorch tensor to call this function'
        return self.data.clone()

    def resize(self, new_size):
        tmp = self.data.detach().clone()
        with torch.no_grad():  del self.data
        self.data = transforms.Resize(new_size)( tmp )
        self.size = new_size

    def save(self, filename):
        self.tens2pil().save(filename)