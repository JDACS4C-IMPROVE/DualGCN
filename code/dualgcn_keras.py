"""
Created by Ruibo. 

Define additional parameters, and Candle Benchmark class in this file. 

"""

import candle
import os

# Just because the tensorflow warnings are a bit verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# This should be set outside as a user environment variable
os.environ['CANDLE_DATA_DIR'] = os.environ['HOME'] + '/improve_data_dir'

# file_path becomes the default location of the example_default_model.txt file
file_path = os.path.dirname(os.path.realpath(__file__))

# Define any needed additional args to ensure all new args are command-line accessible.
additional_definitions = [
    {'name':'n_fold',
     'type':int,
     'nargs':1,
    #  'default': 5,
     'help':'number of folds in the cross validation'
    },

    {'name':'max_atoms',
     'type':int,
     'nargs':1,
    #  'default': 100,
     'help':'max possible size of molecule graphs'
    },

    {'name':'use_gexpr',
     'type':bool,
     'nargs':1,
    #  'default': True,
     'help':'multiomic option'
    },

    {'name':'use_cnv',
     'type':bool,
     'nargs':1,
    #  'default': True,
     'help':'multiomic option'
    },

    {'name':'regression',
     'type':bool,
     'nargs':1,
    #  'default': True,
     'help':'regression or classification'
    },

    # Model structure
    {'name':'drug_gcn_units_list',
     'type': int,
     'nargs': "+", 
    #  'default': [256, 128],
     'help':'Drug GCN units'
    },

    {'name':'cell_feature_fc_units_list',
     'type': int,
     'nargs': "+", 
    #  'default': [32, 128],
     'help':'cell line FC units'
    },

    {'name':'cell_line_gcn_units_list',
     'type': int,
     'nargs': "+", 
    #  'default': [256, 256, 256, 256],
     'help':'cell line GCN units'
    },

    {'name':'universal_dropout',
     'type':float,
     'nargs':1,
    #  'default': 0.1,
     'help':'universal dropout'
    },

    {'name':'fc_layers_dropout',
     'type':float,
     'nargs': "+", 
    #  'default': [0.3, 0.2, 0],
     'help':'the dropout rates for FC layers'
    },

    {'name':'log_dir',
     'type': str,
     'nargs': 1, 
    #  'default': "../log/",
     'help':'dir to save the results files during the evaluation'
    },
]

# Define args that are required.
required = None


# Extend candle.Benchmark to configure the args
class DualGCNBenchmark(candle.Benchmark):

    def set_locals(self):
        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = additional_definitions

