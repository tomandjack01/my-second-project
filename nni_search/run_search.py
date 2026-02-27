#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File Name:     run_search.py
# Author:        Yizhuo Quan
# Created Time:  2023-03-05  09:38
# Last Modified: <none>-<none>

import json
import signal

file = open('search_space.json', 'r')
search_space = json.load(file)

from nni.experiment import Experiment
experiment = Experiment('local')

experiment.config.trial_command = 'python main_graph.py'  # type: ignore[union-attr]
experiment.config.trial_code_directory = '.'  # type: ignore[union-attr]
experiment.config.experiment_working_directory = '.'  # type: ignore[union-attr]


experiment.config.search_space = search_space  # type: ignore[union-attr]
experiment.config.tuner.name = 'TPE'  # type: ignore[union-attr]

experiment.config.tuner.class_args['optimize_mode'] = 'maximize'  # type: ignore[union-attr, index]
experiment.config.max_trial_number = 300  # type: ignore[union-attr]
experiment.config.trial_concurrency = 1  # type: ignore[union-attr]
experiment.run(6006)
if hasattr(signal, 'pause'):
    getattr(signal, 'pause')()  # Unix only
else:
    # Windows: use input to wait
    input('Press Enter to stop the experiment...')
# experiment.stop()
