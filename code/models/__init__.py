import logging
from statistics import mode
logger = logging.getLogger('base')

def create_model(opt):
    model = opt['model']
    
    if model == 'base':
        from.Generation_base import GenerationModel as M
    elif model == 'condition':
        from .Generation_condition import GenerationModel as M
    elif model == "loss":
        from .Generation_loss import GenerationModel as M
    elif model == "gan":
        from .Generation_gan import GenerationModel as M
    elif model == '3img':
        from .Generation_3img import GenerationModel as M
    elif model == 'nolinear':
        from .Generation_nolinear import GenerationModel as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
