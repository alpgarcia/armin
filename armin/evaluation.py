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

from .errors import DatasetError, ResultsError


class NLI4CTEvaluator:
    """Evaluator for the Task 7: Multi-evidence Natural Language
    Inference for Clinical Trial Data (NLI4CT) at SemEval 2023.

    :param gold_path: gold standard file path

    :raises DatasetError: when the gold dataset file path does not exist
        or is not valid
    """

    def __init__(self, gold_path):
        if not os.path.exists(gold_path):
            raise DatasetError(cause="Dataset file %s does not exist" % gold_path)

        # Load gold dataset
        with open(gold_path) as gold_file:
            self.gold = json.load(gold_file)

    def evaluate(self, results_path):
        """Evaluates results against the gold standard

        :param results_path: results file path
        :return:

        :raises ResultsError: when the results file path does not exist
            or is not valid
        """
        if not os.path.exists(results_path):
            raise ResultsError(cause="Results file %s does not exist" % results_path)

        # Load results fle
        with open(results_path) as results_file:
            results = json.load(results_file)

        results_p = []
        gold_p = []
        results_s = []
        gold_s = []

        for uuid in results.keys():
            gold_p.append(self.gold[uuid]["Primary_evidence_index"])
            results_p.append(results[uuid]["Primary_evidence_index"])
            if self.gold[uuid]["Type"] == "Comparison":
                gold_s.append(self.gold[uuid]["Secondary_evidence_index"])
                results_s.append(results[uuid]["Secondary_evidence_index"])

        tp_p, fp_p, fn_p = self.__compare(results_p, gold_p)
        tp_s, fp_s, fn_s = self.__compare(results_s, gold_s)

        tp = tp_p + tp_s
        fp = fp_p + fp_s
        fn = fn_p + fn_s

        if (tp + fp) == 0:
            p_score = 0
        else:
            p_score = tp / (tp + fp)
        r_score = tp / (tp + fn)

        if (p_score + r_score) == 0:
            score = 0
        else:
            score = 2 * (p_score * r_score) / (p_score + r_score)

        return {
            "f1": score,
            "precision": p_score,
            "recall": r_score
        }

    @staticmethod
    def __compare(results_ev, gold_ev):
        tp = 0
        fp = 0
        fn = 0
        for i in range(len(gold_ev)):
            for j in range(len(results_ev[i])):
                if results_ev[i][j] in gold_ev[i]:
                    tp += 1
                if results_ev[i][j] not in gold_ev[i]:
                    fp += 1
            for j in range(len(gold_ev[i])):
                if gold_ev[i][j] not in results_ev[i]:
                    fn += 1

        return tp, fp, fn
