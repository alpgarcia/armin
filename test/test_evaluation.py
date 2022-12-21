# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Alberto Pérez García-Plaza
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors:
#     Alberto Pérez García-Plaza <alberto.perez@lsi.uned.es>
#

import json
import os
import sys
import unittest

# Make sure we use our code and not any other could we have installed
sys.path.insert(0, '..')

from armin.evaluation import NLI4CTEvaluator


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.__data_dir = \
            os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         'data')

    def test_evaluate_bm25_baseline_results(self):
        """Test the evaluation of the baseline results"""
        bm_25_results_path = os.path.join(self.__data_dir,
                                          'results_bm25.json')

        ev = NLI4CTEvaluator(os.path.join(self.__data_dir,
                                          'gold_dev.json'))

        self.assertEqual(ev.evaluate(bm_25_results_path),
                         {
                             "f1": 0.322747893713545,
                             "precision": 0.42203389830508475,
                             "recall": 0.26128016789087094
                         })

