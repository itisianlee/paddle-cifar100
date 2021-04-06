from paddle.optimizer import Momentum, AdamW, Adam, SGD

optim_dict = {
    'momentum': Momentum,
    'adamw': AdamW,
    'adam': Adam,
    'sgd': SGD,
}

def build_optim(cfg, parameters=None, learning_rate=0.001):
    """build optimizer
    Args:
        cfg : A CfgNode instance.
        learning_rate (float|LRScheduler): The learning rate used to update ``Parameter``.
            It can be a float value or any subclass of ``LRScheduler`` .
        parameters (list, optional): List of ``Tensor`` names to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
    Returns:
       A optimizer.
    """
    if cfg.name not in list(optim_dict.keys()):
        raise KeyError(f'The optimizer must be one of the following lists: momentum|adamw|adam|sgd, but get {cfg.name}')
    optim_class = optim_dict.get(cfg.name)
    return optim_class(parameters=parameters, learning_rate=learning_rate, **cfg.params)