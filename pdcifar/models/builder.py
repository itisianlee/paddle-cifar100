from ..utils import Registry

classifier = Registry('classifier')

def build_classifier(name):
    if name not in classifier:
        raise KeyError(f'{name} is not in the {classifier.name} registry')
    net = classifier.get(name)
    return net()