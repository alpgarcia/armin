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
import datetime
import json
import os
import time

from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi

from armin.errors import CTRError, DatasetError


class CTRReader:

    def __init__(self, ctrs_folder_path):
        if not os.path.exists(ctrs_folder_path):
            raise CTRError(cause="CTRS folder %s does not exist"
                                 % ctrs_folder_path)

        self.ctrs_folder_path = ctrs_folder_path

    def read_section(self, ctr_id, section_id):
        """ Reads a CTR section given their identifiers

        :param ctr_id: CTR identifier
        :param section_id: identifier of the CTR section to read

        :raises CTRError: when the CTR file does not exist or is not
        valid
        """
        ctr_path = os.path.join(self.ctrs_folder_path, ctr_id + ".json")

        if not os.path.exists(ctr_path):
            raise CTRError(cause="CTRS file %s does not exist" % ctr_path)

        with open(ctr_path) as ctr_file:
            ctr = json.load(ctr_file)

        return ctr[section_id]


class Baseline:
    """Baseline evidence extractor based on BM25 for
    NLI4CT Semeval'23 task.

    :param dataset_path: path of the dataset
    :param ctrs_folder_path: path to the folder containing
        the set of CTR files

    :raises DatasetError: when the dataset file path does not exist
        or is not valid
    :raises CTRError: when the CTRs folder does not exist or is not
        valid
    """

    def __init__(self, dataset_path, ctrs_folder_path):

        if not os.path.exists(dataset_path):
            raise DatasetError(cause="Dataset file %s does not exist"
                                     % dataset_path)

        self._crt_reader = CTRReader(ctrs_folder_path)

        # Load dataset
        with open(dataset_path) as dataset_file:
            self.dataset = json.load(dataset_file)

        self._tokenizer = RegexpTokenizer(r"[\w']+")

        try:
            self._stops_en = set(stopwords.words('english'))
        except LookupError:
            import nltk
            nltk.download('stopwords')
            self._stops_en = set(stopwords.words('english'))

    def tokenize(self, text):
        """ Tokenize a text

        :param text: text to be tokenized
        """

        # tokenized = [x.split(" ") for x in section]
        # tokenized = [[x.strip(' ') for x in y] for y in tokenized]
        # tokenized = [[x for x in y if x] for y in tokenized]

        tokens = self._tokenizer.tokenize(text.lower())

        # Remove stopwords from text using nltk stopwords
        tokens = [t for t in tokens if t not in self._stops_en]

        return tokens

    def _retrieve_section_evidences(self, statement_tokens, section):
        """Search evidences for the given statement within the
         given section.

        :param statement_tokens: statement tokens to be used as query to look
            for evidences
        :param section: CTR section to extract evidences from

        :return: list of indexes of those sentences in the section
            considered as evidences for the given statement
        """
        # create an instance of the BM25 class, which reads in the
        # section tokenized text and does some indexing on it
        tokenized = [self.tokenize(sent) for sent in section]
        bm25 = BM25Okapi(tokenized)
        # Retrieve bm25 scores for the primary section using
        # the tokenized statement as a query
        scores = bm25.get_scores(statement_tokens)
        # Retrieve all entries from the primary section with a bm25
        # score over 1
        section_evidences = [i for i in range(len(scores))
                             if scores[i] > 1]

        return section_evidences

    def retrieve_evidences(self):
        """Retrieve evidences for each statement in the dataset from the
        corresponding CTR files and sections.

        :return: a dict with the primary and secondary (when applies)
            evidences retrieved from the CTRs. Evidences are represented
            as a list of indexes of each sentence considered as evidence
            for the given statement.
        """
        total = len(self.dataset.keys())
        completed = 0
        start = time.perf_counter()
        elapsed = 0
        results = {}
        for uuid in self.dataset.keys():
            ps_info = self.dataset[uuid]
            statement_tokens = self.tokenize(ps_info["Statement"])
            primary_section = \
                self._crt_reader.read_section(ps_info["Primary_id"],
                                              ps_info["Section_id"])

            primary_evidences = self._retrieve_section_evidences(
                    statement_tokens,
                    primary_section)
            results[uuid] = {"Primary_evidence_index": primary_evidences}

            # Repeat for the secondary trial
            if ps_info["Type"] == "Comparison":
                secondary_section = self._crt_reader.read_section(
                        ps_info["Secondary_id"],
                        ps_info["Section_id"])

                secondary_evidences = self._retrieve_section_evidences(
                        statement_tokens,
                        secondary_section)
                results[uuid]["Secondary_evidence_index"] = secondary_evidences

            completed += 1
            end = time.perf_counter()
            elapsed = (end - start)
            estimated = (1 / (completed / total)) * elapsed
            print(f"\r[{uuid}] Completion: {completed}/{total} "
                  f"Elapsed: {datetime.timedelta(seconds=round(elapsed))} - "
                  f"Estimated: {datetime.timedelta(seconds=round(estimated))}",
                  end='')

        print(f"\rCompletion: {completed}/{total} "
              f"Elapsed: {datetime.timedelta(seconds=round(elapsed))}")
        return results
