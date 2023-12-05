# -*- coding: utf-8 -*-
#
# Copyright (C) 2023 Alberto Pérez García-Plaza
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

from .errors import DatasetError
from .extractors import CTRReader


class JSONTransformer:
    """Transforms a dataset in JSON format (NLI4CT Semeval'23 task)
    into a set of pairs of sentences (statement, evidence) and their
    relationship (entailment, contradiction, neutral).

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

    def transform(self, out_file):
        """Transforms the dataset into a set of pairs of sentences
        (statement, evidence) and their relationship (entailment,
        contradiction, neutral).

        :param out_file: output file path
        """
        with open(out_file, "w") as out:
            out.write("[\n")
            for uuid in self.dataset.keys():
                # get premise-statement info
                ps_info = self.dataset[uuid]

                primary_section = \
                    self._crt_reader.read_section(ps_info["Primary_id"],
                                                  ps_info["Section_id"])
                self._write_pairs(uuid,
                                  ps_info["Type"],
                                  ps_info["Section_id"],
                                  ps_info["Primary_evidence_index"],
                                  ps_info["Statement"],
                                  primary_section,
                                  ps_info["Label"],
                                  out)

                if ps_info["Type"] == "Comparison":
                    secondary_section = \
                        self._crt_reader.read_section(ps_info["Secondary_id"],
                                                      ps_info["Section_id"])
                    self._write_pairs(uuid,
                                      ps_info["Type"],
                                      ps_info["Section_id"],
                                      ps_info["Secondary_evidence_index"],
                                      ps_info["Statement"],
                                      secondary_section,
                                      ps_info["Label"],
                                      out)
            out.write("]\n")

    @staticmethod
    def _write_pairs(uuid, _type, section_id,
                     evidence_indexes, statement, sentences, label,
                     out):
        """Writes a pair of sentences (statement, evidence), their
        relationship (entailment, contradiction, neutral) and some
        other info useful to characterize the Premise-Statement pair
        to the output file.

        :param uuid: unique identifier of the Premise-Statement pair
        :param _type: Single or Comparison
        :param section_id: identifier of the section from which the evidences
            are extracted
        :param evidence_indexes: indexes of the evidences in the section
        :param statement: statement
        :param sentences: list of sentences to write pairs from
        :param label: relationship between the statement and the
            evidence (entailment | contradiction)
        :param out: output file
        """
        sentence_number = 0
        for sent in sentences:
            # For each sentence, mark it as neutral if its index is not
            # included in the evidence indexes set, use `label` value otherwise
            #
            # relationship = entailment | contradiction | neutral
            if sentence_number in evidence_indexes:
                relationship = label.lower()
            else:
                relationship = "neutral"

            out.write(json.dumps({
                "uuid": uuid,
                "section_id": section_id,
                "type": _type,
                "target": relationship,
                "statement": statement,
                "evidence": sent

            }) + ",\n")

            sentence_number += 1
