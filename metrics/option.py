import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from collections import OrderedDict

def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper
Loader, Dumper = OrderedYaml()
def parse(opt_path):
    with open(opt_path,mode='r') as f :
        opt = yaml.load(f, Loader=Loader)
    
    return opt


