#!/usr/bin/env python3
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
from typing import Callable

from armin.evaluation import NLI4CTEvaluator
from armin.extractors.baseline import Baseline
from armin.extractors.ontobiosim import OntobioSim
from armin.extractors.semanticsim import PassageRankingSim, SemanticSearch, SemanticSearchPubmed

DATASET_PATH = "./training_data/train.json"
BASELINE_RESULTS_PATH = "./results/baseline_results.json"
ONTOBIO_RESULTS_PATH = "./results/ontobio_results.json"
PASSAGE_RANKING_RESULTS_PATH = "./results/pasranking_results.json"
SEMANTIC_SEARCH_RESULTS_PATH = "./results/ssearch_results.json"
SEMANTIC_SEARCH_PM_RESULTS_PATH = "./results/ssearch_pm_results.json"


def run_baseline():
    print("\nExecuting baseline experiment\n")
    baseline = Baseline(DATASET_PATH, "./training_data/CT json")
    evidences = baseline.retrieve_evidences()
    # print(evidences)
    print("Writing results to " + BASELINE_RESULTS_PATH)
    with open(BASELINE_RESULTS_PATH, 'w') as jsonFile:
        jsonFile.write(json.dumps(evidences, indent=4))
    # Evaluate results
    evaluator = NLI4CTEvaluator(DATASET_PATH)
    metrics = evaluator.evaluate(BASELINE_RESULTS_PATH)
    print('BM25 F1:{:f}'.format(metrics["f1"]))
    print('BM25 precision_score:{:f}'.format(metrics["precision"]))
    print('BM25 recall_score:{:f}'.format(metrics["recall"]))


def run_ontobio():
    print("\nExecuting Ontobio experiment:\n")
    ob_sim = OntobioSim(DATASET_PATH, "./training_data/CT json", 0.1)
    evidences = ob_sim.retrieve_evidences()
    # print(evidences)
    print("Writing results to " + ONTOBIO_RESULTS_PATH)
    with open(ONTOBIO_RESULTS_PATH, 'w') as jsonFile:
        jsonFile.write(json.dumps(evidences, indent=4))
    # Evaluate results
    evaluator = NLI4CTEvaluator(DATASET_PATH)
    metrics = evaluator.evaluate(ONTOBIO_RESULTS_PATH)
    print('Ontobio F1:{:f}'.format(metrics["f1"]))
    print('Ontobio precision_score:{:f}'.format(metrics["precision"]))
    print('Ontobio recall_score:{:f}'.format(metrics["recall"]))


def run_passage_ranking(threshold: float) -> dict:
    prsim = PassageRankingSim(DATASET_PATH,
                              "./training_data/CT json",
                              threshold=threshold)
    evidences = prsim.retrieve_evidences()
    # print(evidences)
    # print("Writing results to " + PASSAGE_RANKING_RESULTS_PATH)
    with open(PASSAGE_RANKING_RESULTS_PATH, 'w') as jsonFile:
        jsonFile.write(json.dumps(evidences, indent=4))
    # Evaluate results
    evaluator = NLI4CTEvaluator(DATASET_PATH)
    _metrics = evaluator.evaluate(PASSAGE_RANKING_RESULTS_PATH)
    # print('PRSim F1:{:f}'.format(_metrics["f1"]))
    # print('PRSim precision_score:{:f}'.format(_metrics["precision"]))
    # print('PRSim recall_score:{:f}'.format(_metrics["recall"]))

    return _metrics


def run_semantic_search(threshold: float) -> dict:
    """Run Semantic Search experiment."""

    ssim = SemanticSearch(DATASET_PATH,
                          "./training_data/CT json",
                          threshold=threshold)
    evidences = ssim.retrieve_evidences()
    # print(evidences)
    # print("Writing results to " + SEMANTIC_SEARCH_RESULTS_PATH)
    with open(SEMANTIC_SEARCH_RESULTS_PATH, 'w') as jsonFile:
        jsonFile.write(json.dumps(evidences, indent=4))
    # Evaluate results
    evaluator = NLI4CTEvaluator(DATASET_PATH)
    _metrics = evaluator.evaluate(SEMANTIC_SEARCH_RESULTS_PATH)

    # print('SSim F1:{:f}'.format(_metrics["f1"]))
    # print('SSim precision_score:{:f}'.format(_metrics["precision"]))
    # print('SSim recall_score:{:f}'.format(_metrics["recall"]))

    return _metrics


def run_semantic_search_pubmed(threshold: float) -> dict:
    """Run Semantic Search experiment."""

    ssimpm = SemanticSearchPubmed(DATASET_PATH,
                                  "./training_data/CT json",
                                  threshold=threshold)
    evidences = ssimpm.retrieve_evidences()

    with open(SEMANTIC_SEARCH_PM_RESULTS_PATH, 'w') as jsonFile:
        jsonFile.write(json.dumps(evidences, indent=4))
    # Evaluate results
    evaluator = NLI4CTEvaluator(DATASET_PATH)
    _metrics = evaluator.evaluate(SEMANTIC_SEARCH_PM_RESULTS_PATH)

    print('SSimPM F1:{:f}'.format(_metrics["f1"]))
    print('SSimPM precision_score:{:f}'.format(_metrics["precision"]))
    print('SSimPM recall_score:{:f}'.format(_metrics["recall"]))

    return _metrics


def find_best_threshold(extractor: Callable[[float], dict],
                        start: float, end: float, step: float) -> float:
    """Find the best threshold for the given extractor."""

    results = []
    t = start
    while t <= end:
        print("Testing threshold: " + str(t))
        metrics = extractor(t)
        results.append((t, metrics["f1"]))
        t += step

    results.sort(key=lambda tup: tup[1], reverse=True)
    print("\n------\nRESULTS:\n-------")
    for t, f1 in results:
        print("Threshold: " + str(t) + " F1: " + str(f1))

    return results[0][0]


if __name__ == '__main__':
    run_baseline()

    # run_ontobio()

    # Test different thresholds for the Passage Ranking extractor.
    # Present the thresholds based on f1 results
    # print("\nExecuting Passage Ranking experiment\n")
    # print("Finding best threshold...")
    # best_t = find_best_threshold(run_passage_ranking,
    #                              10, 150, 10)
    # print("Best threshold: " + str(best_t))

    # Test different thresholds for the semantic search extractor.
    # Present the thresholds based on f1 results
    # print("\nExecuting Semantic Search experiment\n")
    # print("Finding best threshold...")
    # best_t = find_best_threshold(run_semantic_search,
    #                              0.1, 0.9, 0.1)
    # print("Best threshold: " + str(best_t))

    print("\nExecuting Semantic Search Pubmed experiment\n")
    print("Finding best threshold...")
    best_t = find_best_threshold(run_semantic_search_pubmed,
                                 0.1, 0.9, 0.1)
    print("Best threshold: " + str(best_t))
