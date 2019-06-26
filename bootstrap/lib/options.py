import os
import sys
import yaml
import json
import copy
import inspect
import argparse
import collections
from collections import OrderedDict
from yaml import Dumper
from bootstrap.lib.utils import merge_dictionaries


class Options(object):

    def __init__(self, path_yaml=None):
        self.options= Options.load_yaml_opts(path_yaml)


    def __getitem__(self, key):
        """
        """
        val = self.options[key]
        return val


    def __setitem__(self, key, val):
        self.options[key] = val


    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            return object.__getattr__(self, key)


    def __contains__(self, item):
        return item in self.options


    def __str__(self):
        return json.dumps(self.options, indent=2)


    def get(self, key, default):
        return self.options.get(key, default)


    def copy(self):
        return self.options.copy()


    def has_key(self, k):
        return k in self.options


    def keys(self):
        return self.options.keys()


    def values(self):
        return self.options.values()


    def items(self):
        return self.options.items()


    def save(self, path_yaml):
        """ Write options dictionary to a yaml file
        """
        Options.save_yaml_opts(self.options, path_yaml)


    # static methods
    def load_yaml_opts(path_yaml):
        """ Load options dictionary from a yaml file
        """
        result = {}
        print("debug",path_yaml)
        with open(path_yaml, 'r') as yaml_file:
            options_yaml = yaml.load(yaml_file,Loader=yaml.FullLoader)
            includes = options_yaml.get('__include__', False)
            if includes:
                if type(includes) != list:
                    includes = [includes]
                for include in includes:
                    parent = Options.load_yaml_opts('{}/{}'.format(os.path.dirname(path_yaml), include))
                    merge_dictionaries(result, parent)
            merge_dictionaries(result, options_yaml)  # to be sure the main options overwrite the parent options
        result.pop('__include__', None)
        return result

    def save_yaml_opts(opts, path_yaml):
        # Warning: copy is not nested
        options = copy.copy(opts)
        if 'path_opts' in options:
            del options['path_opts']


        with open(path_yaml, 'w') as yaml_file:
            yaml.dump(options, yaml_file, Dumper=Dumper, default_flow_style=False)
