
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


from tqdm import tqdm


class TrainMethod():

    def __init__(self, optimizer, logger = 'tqdm', logger_period : int = 20):

        self.optimizer = optimizer

        self.logger = logger
        self.logger_period = logger_period

        self.iter_count = 0
        self.closure = None
        self.pbar = None # to use with tqdm


    def set_closure(self, closure):
        self.closure = closure


    def loop(self, iter_max, desc : str = '', auto_increment : bool = False, auto_logger : bool = False):

        self.iter_count = 0       # reset counter to zero
        self.iter_max = iter_max  # to make it accessible outside

        if self.logger == 'tqdm':
            # use the tqdm progress bar
            self.pbar = tqdm(total=iter_max, desc=desc)


        while self.iter_count <= iter_max:
            self.optimizer.step(self.closure)
            
            if auto_increment: self.iter_count += 1
            if auto_logger:    self.logger_update()


    def logger_update(self, msg : str = ''):
        if self.logger_should_update():
            if self.logger == 'tqdm':
                self.pbar.update( self.logger_period )
                if msg != '':  self.pbar.set_postfix_str(msg)

            else:
                print("{}/{}\t".format(self.iter_count, self.max_iter) + msg)


    def logger_should_update(self):
        return (self.iter_count%self.logger_period == 0)