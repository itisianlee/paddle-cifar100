from paddle.optimizer.lr import CosineAnnealingDecay, MultiStepDecay, LinearWarmup


sch_dict = {'step': MultiStepDecay,
            'cosine': CosineAnnealingDecay}

def build_lrscheduler(cfg):
    """build lr scheduler
    Args:
        cfg : A CfgNode instance.
    Returns:
       A lr scheduler.
    """
    if cfg.name not in list(sch_dict.keys()):
        raise KeyError(f'The optimizer must be one of the following lists: step|cosine, but get {cfg.name}')
    sch_class = sch_dict.get(cfg.name)
    assert cfg.warm_up_step >= 0, f'Warm Up step has a value greater than or equal to 0, but get {cfg.warm_up_step}'
    if cfg.warm_up_step > 0:
        return LinearWarmup(sch_class(**cfg.params), cfg.warm_up_step, 0., cfg.params.learning_rate)
    else:
        return sch_class(**cfg.params)