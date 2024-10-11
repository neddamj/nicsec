class Compressor:
    """
        I need this to be an interface through which I can input a neural compression model 
        or a classical compression model (JPEG etc) and have it spit out the relevant outputs
        so that I can calculate the gradients of the inputs w.r.t the loss function.

    """
    def __init__(self):
        pass

    def __call__(self, *args, **kwds):
        pass