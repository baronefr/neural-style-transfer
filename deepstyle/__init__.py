
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


__version__ = '1.0.0'
__major_review__ = '6 Feb 2023'


import deepstyle

import deepstyle.models
import deepstyle.tools

from deepstyle.image import ImageManager


####################
#  easy interface  #
####################

def version():
    print('deepstyle | v', __version__)
    print(' major review:', __major_review__)

def credits():
    print('deepstyle | v', __version__)
    print(' F.P. Barone, D. Ninni, P. Zinesi')
    print(' github.com/baronefr')
