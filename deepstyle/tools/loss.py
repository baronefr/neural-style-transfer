
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
 
 
import torch
import torch.nn as nn



class LossManager():
    """This class collects all the useful stuff to compute & track the losses"""
    def __init__(self, layers : list, targets : list,
                 lossfn = lambda x: x,   # loss function to call for all values in this collection
                 weights : list | None = None,  # if provided, will use the given weights
                 scale : float = 1.0     # general weight scale factor 
                ):
        assert len(targets) == len(layers)

        n = len(layers)

        self.layers = layers
        self.targets = targets
        self.lossfn = lossfn

        if weights is None:
            self.weights = [ 1/n ]*n
        else:
            assert len(weights) == n
            self.weights = weights

        self.hist = []
        self.scale = scale

    def compute_loss(self, feats : list):
        tmp = self.scale*sum([ self.weights[ii]*self.lossfn(ft, self.targets[ii]) for ii, ft in enumerate(feats) ])

        self.hist.append( tmp.item() )
        return tmp
    
    def compute_loss_from_dict(self, feats : dict):
        feats = [ feats[key] for key in self.layers ]
        
        tmp = self.scale*sum([ self.weights[ii]*self.lossfn(ft, self.targets[ii]) for ii, ft in enumerate(feats) ])
        self.hist.append( tmp.item() )
        return tmp
    

# compute a gram matrix from tensor
def gramfn( xx : torch.Tensor , use_batch : bool = True):
    if use_batch:
        batch, ch, H, W = xx.size()
    else:
        ch, H, W = xx.size()
        batch = 1
    view = xx.view(batch, ch, H*W)
    gram = torch.matmul(view, view.transpose(1,2))
    gram.div_(H*W)
    return gram


# I wrap it to look like a loss function from pytorch
class GramLoss( nn.Module ):
    def forward(self, input, target):
        # note: target should not require gradient!
        #       and must be a gram matrix
        return nn.MSELoss()( gramfn(input), target )
    
