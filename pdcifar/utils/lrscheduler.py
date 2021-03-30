from paddle.optimizer.lr import CosineAnnealingDecay, MultiStepDecay, LinearWarmup


sch_dict = {'step': MultiStepDecay,
            'cosine': CosineAnnealingDecay}

def build_lrscheduler(name=None, learning_rate=0.001, warm_up_step=2000, **kwargs):
    """build lr scheduler
    Args:
        learning_rate (float): The learning rate used to update ``Parameter``.
        warm_up_step (int): Warm Up step has a value greater than or equal to 0.
    Returns:
       A lr scheduler.
    """
    if name is None:
        return learning_rate
    if name not in list(sch_dict.keys()):
        raise KeyError(f'The optimizer must be one of the following lists: step|cosine, but get {name}')
    sch_class = sch_dict.get(name)
    assert warm_up_step >= 0, f'Warm Up step has a value greater than or equal to 0, but get {warm_up_step}'
    if warm_up_step > 0:
        return LinearWarmup(sch_class(learning_rate=learning_rate, **kwargs), warm_up_step, 0., learning_rate)
    else:
        return sch_class(learning_rate=learning_rate, **kwargs)