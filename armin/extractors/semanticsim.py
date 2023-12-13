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
import time

from sentence_transformers import SentenceTransformer, util
from torch import Tensor

from armin.extractors.baseline import Baseline

# sentences = ["I'm very happy", "I'm happy"]
#
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#
# #Compute embedding for both lists
# embedding_1 = model.encode(sentences[0], convert_to_tensor=True)
# embedding_2 = model.encode(sentences[1], convert_to_tensor=True)
#
# print(util.pytorch_cos_sim(embedding_1, embedding_2))
#
# source_sentence = "That is a happy person"
# sentences2 = ["That is a happy dog",
#               "That is a very happy person",
#               "Today is a sunny day"]
#
# model2 = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
#
# embedding_3 = model2.encode(source_sentence, convert_to_tensor=True)
# embedding_4 = model2.encode(sentences2[0], convert_to_tensor=True)
# embedding_5 = model2.encode(sentences2[1], convert_to_tensor=True)
# embedding_6 = model2.encode(sentences2[2], convert_to_tensor=True)
#
# print(util.pytorch_cos_sim(embedding_3, embedding_4))
# print(util.pytorch_cos_sim(embedding_3, embedding_5))
# print(util.pytorch_cos_sim(embedding_3, embedding_6))


class PassageRankingSim(Baseline):
    """Passage Ranking based evidence extractor.

        :param dataset_path: path of the dataset
        :param ctrs_folder_path: path to the folder containing
            the set of CTR files
        :param threshold: similarity threshold to decide whether a
            sentence is an evidence or not. Defaults to 80

        :raises DatasetError: when the dataset file path does not exist
            or is not valid
        :raises CTRError: when the CTRs folder does not exist or is not
            valid
        """

    def __init__(self, dataset_path, ctrs_folder_path, threshold=80):
        super().__init__(dataset_path, ctrs_folder_path)

        self._threshold = threshold

        self._init_model()

    def _init_model(self):
        print("Creating model...", end='')
        self._model = SentenceTransformer(
                'sentence-transformers/msmarco-distilbert-base-tas-b')
        print("Done!")

    def _compute_similarity(self,
                            st_embedding,
                            sect_sentence_embedding) -> Tensor:
        # TODO compute similarity for a whole section at once
        return util.dot_score(st_embedding, sect_sentence_embedding)

    def _retrieve_section_evidences(self, st_embedding, section):
        """Search evidences for the given statement within the
         given section.

        :param st_embedding: statement embedding to be used as query
        :param section: CTR section to extract evidences from

        :return: list of indexes of those sentences in the section
            considered as evidences for the given statement
        """
        scores = []
        for sect_sentence in section:
            sect_sentence_embedding = \
                self._model.encode(sect_sentence, convert_to_tensor=True)
            score: Tensor = self._compute_similarity(st_embedding,
                                                     sect_sentence_embedding)
            scores.append(score.item())

        section_evidences = [i for i in range(len(scores))
                             if scores[i] > self._threshold]

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
            st_embedding = self._model.encode(ps_info["Statement"],
                                              convert_to_tensor=True)
            primary_section = \
                self._crt_reader.read_section(ps_info["Primary_id"],
                                              ps_info["Section_id"])

            primary_evidences = self._retrieve_section_evidences(
                    st_embedding,
                    primary_section)
            results[uuid] = {"Primary_evidence_index": primary_evidences}

            # Repeat for the secondary trial
            if ps_info["Type"] == "Comparison":
                secondary_section = self._crt_reader.read_section(
                        ps_info["Secondary_id"],
                        ps_info["Section_id"])

                secondary_evidences = self._retrieve_section_evidences(
                        st_embedding,
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


class SemanticSearch(PassageRankingSim):
    """Semantic Search based evidence extractor.

        :param dataset_path: path of the dataset
        :param ctrs_folder_path: path to the folder containing
            the set of CTR files
        :param threshold: similarity threshold to decide whether a
            sentence is an evidence or not. From 0 to 1, defaults to 0.5

        :raises DatasetError: when the dataset file path does not exist
            or is not valid
        :raises CTRError: when the CTRs folder does not exist or is not
            valid
        """

    def __init__(self, dataset_path, ctrs_folder_path, threshold=.5):
        super().__init__(dataset_path, ctrs_folder_path, threshold)

    def _init_model(self):
        print("Creating model...", end='')
        # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        self._model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2')
        print("Done!")

    def _compute_similarity(self,
                            st_embedding,
                            sect_sentence_embedding) -> Tensor:
        return util.pytorch_cos_sim(st_embedding, sect_sentence_embedding)


class SemanticSearchPubmed(PassageRankingSim):
    """Semantic Search based evidence extractor.

        :param dataset_path: path of the dataset
        :param ctrs_folder_path: path to the folder containing
            the set of CTR files
        :param threshold: similarity threshold to decide whether a
            sentence is an evidence or not. From 0 to 1, defaults to 0.5

        :raises DatasetError: when the dataset file path does not exist
            or is not valid
        :raises CTRError: when the CTRs folder does not exist or is not
            valid
        """

    def __init__(self, dataset_path, ctrs_folder_path, threshold=.5):
        super().__init__(dataset_path, ctrs_folder_path, threshold)

    def _init_model(self):
        print("Creating model...", end='')
        self._model = SentenceTransformer(
                'tavakolih/all-MiniLM-L6-v2-pubmed-full')
        print("Done!")

    def _compute_similarity(self,
                            st_embedding,
                            sect_sentence_embedding) -> Tensor:
        return util.pytorch_cos_sim(st_embedding, sect_sentence_embedding)