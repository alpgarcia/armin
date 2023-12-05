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

from ontobio import OntologyFactory

from armin.extractors.baseline import Baseline


class OntobioSim(Baseline):
    """Ontobio based evidence extractor.

        :param dataset_path: path of the dataset
        :param ctrs_folder_path: path to the folder containing
            the set of CTR files
        :param threshold: similarity threshold to decide whether a
            sentence is an evidence or not. From 0 to 1, defaults to 0.2

        :raises DatasetError: when the dataset file path does not exist
            or is not valid
        :raises CTRError: when the CTRs folder does not exist or is not
            valid
        """

    def __init__(self, dataset_path, ctrs_folder_path, threshold=.2):
        super().__init__(dataset_path, ctrs_folder_path)

        self._threshold = threshold

        # Create ontology object, for GO
        # Transparently uses remote SPARQL service.
        # (May take a few seconds to run first time)
        print("Creating ontology factory...", end='')
        o_factory = OntologyFactory()
        self.__ont = o_factory.create('go')
        print("Done!")

    def _retrieve_section_evidences(self, statement_tokens, section):
        """Search evidences for the given statement within the
         given section.

        :param statement_tokens: statement tokens to be used as query to look
            for evidences
        :param section: CTR section to extract evidences from

        :return: list of indexes of those sentences in the section
            considered as evidences for the given statement
        """
        statement_ents = []
        for term in statement_tokens:
            ids = self.__ont.search(term + '%')
            if ids:
                statement_ents.append(ids)

        # flatten list
        statement_ents = \
            [item for sublist in statement_ents for item in sublist]

        # compute similarity for each sentence
        section_evidences = []
        index = 0
        for sentence in section:
            sentence_ents = []
            for term in self.tokenize(sentence):
                ids = self.__ont.search(term + '%')
                if ids:
                    sentence_ents.append(ids)

            # flatten list
            sentence_ents = \
                [item for sublist in sentence_ents for item in sublist]

            common = len(set(statement_ents) & set(sentence_ents))
            if common == 0:
                continue
            sim = common / (len(statement_ents) + len(sentence_ents))
            if sim > self._threshold:
                section_evidences.append(index)
            index += 1

        return section_evidences
