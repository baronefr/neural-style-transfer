#########################################################
#   DEEPSTYLE
#
#   neural networks and deep learning exam project
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  coder: Barone Francesco, last edit: 4 Feb 2023
#--------------------------------------------------------


__version__ = '0.1.0'
__major_review__ = '3 Feb 2023'


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
