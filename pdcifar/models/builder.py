from ..utils import Registry

classifier = Registry('classifier')

def build_classifier(cfg):
    if cfg.name not in classifier:
        raise KeyError(f'{cfg.name} is not in the {classifier.name} registry')
    net = classifier.get(cfg.name)
    if 'params' in cfg:
        return net(**cfg.params)
    else:
        return net()