from argparse import Namespace

import yaml


def load_args(config_path, profile):
    with open(config_path, encoding='utf-8') as f:
        args = yaml.load(f, Loader=yaml.Loader)[profile]
    args = Namespace(**args['model'], **args['data'], **args['optim'],
                     **args['misc'])
    return args
