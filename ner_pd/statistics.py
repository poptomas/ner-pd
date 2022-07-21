import json
from abc import ABC
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import math

from .parsers import StatisticsExperimentParser
from .filename import Filename


@dataclass
class StatInfo:
    model: str
    entity: str
    sentences: int
    occurrence_sum: int


def get_entity_types_ontonotes():
    return [
        "CARDINAL",
        "DATE",
        "EVENT",
        "FAC",
        "GPE",
        "LANGUAGE",
        "LAW",
        "LOC",
        "MONEY",
        "NORP",
        "ORDINAL",
        "ORG",
        "PERCENT",
        "PERSON",
        "PRODUCT",
        "QUANTITY",
        "TIME",
        "WORK_OF_ART",
    ]


class Statistics(ABC):
    """
    Common ground for statistics-based experiments conducted
    during the writing of the thesis
    """

    def __init__(self):
        self.positive_emails_data = {}
        self.negative_emails_data = {}
        self.stats = []

    def load_data(self, output_file: str, args: StatisticsExperimentParser):
        self.args = args
        self.outfile = output_file
        self.positive_emails_data = self.open_file(args.positive)
        self.negative_emails_data = self.open_file(args.negative)
        self.file = Filename(relpath=args.outdir, name=output_file, extension=".txt")

    def open_file(self, filename: str):
        with open(filename, "r") as file:
            return json.load(file)


class Divergence(Statistics):
    """ """

    def run(self):
        """
        Iterates through neg/pos emails datasets provided in "data" directory,
        each entity type undergoes a divergence computation,
        obtained data are saved as a JSON file further used for plot generation
        """

        entity_types = get_entity_types_ontonotes()  # OntoNotes v5 only
        for (_, h_value), (_, s_value) in zip(
            self.positive_emails_data.items(), self.negative_emails_data.items()
        ):
            data = {}
            for ent in entity_types:
                if self.args.occurrences:
                    data[ent] = self.__run_with_entity_occurrences(
                        ent, h_value, s_value
                    )
                else:
                    data[ent] = self.__run_entity_in_sentence(ent, h_value, s_value)
            fname = self.file.get_without_extension()
            self.__write_json(data, fname)

    def __kl_divergence(self, first_dist: list[float], second_dist: list[float], base):
        """
        Kullback-Leibler Divergence computation
        - https://machinelearningmastery.com/divergence-between-probability-distributions/
        """

        cumulative_sum = 0
        for idx in range(len(first_dist)):
            cumulative_sum += (
                first_dist[idx]
                * np.log(first_dist[idx] / second_dist[idx])
                / np.log(base)
            )
        return cumulative_sum

    def __js_divergence(
        self, first: list[float], second: list[float], common: list[float], base
    ):
        """
        Jensen-Shannon Divergence computation
        - https://machinelearningmastery.com/divergence-between-probability-distributions/
        """

        return 0.5 * self.__kl_divergence(
            first, common, base
        ) + 0.5 * self.__kl_divergence(second, common, base)

    def __key_union(
        self,
        first: list[float],
        second: list[float],
        select=lambda first, second: (first, second),
    ) -> dict:
        """
        Key union to ensure that entity types of the dataset pairs do not differ
        - assuming a small dataset, it may happen that not a single record of a given entity type
        was found, and the second one contained it
        - moreover, correction is introduced to avoid zero division and log 0 in KL divergence computation
        """
        correction = 1
        # print(first, second)
        return {
            key: select(first.get(key, 0) + correction, second.get(key, 0) + correction)
            for key in first.keys() | second.keys()
        }

    def __get_entity_distribution_values(self, entity: str, dictionary: dict) -> dict:
        """
        Compute how many records make up the distribution by adding
        not-found key subtracting the number of sentences and the "at least once found in the sentence" named entities
        """
        alternative = {"0": dictionary["sentences"]}
        distribution = dictionary.get(entity, alternative)
        if distribution != alternative:
            distribution["0"] = dictionary["sentences"] - sum(distribution.values())
        return distribution

    def __get_normalized_dictionary(
        self, dictionary: dict, sum_v: int, index: int
    ) -> dict:
        return {key: value[index] / sum_v for key, value in dictionary.items()}

    def __get_normalized_values(self, dictionary: dict) -> Tuple[dict, dict]:
        """
        - obtain distributions from the named entity occurrences dictionary
        - note: it does not have to necessarily be a positive-negative pair, as shown in the method
        """
        positive_emails_sum = 0
        negative_emails_sum = 0
        for pos_v, neg_v in dictionary.values():
            positive_emails_sum += pos_v
            negative_emails_sum += neg_v
        pos_dict = self.__get_normalized_dictionary(dictionary, positive_emails_sum, 0)
        neg_dict = self.__get_normalized_dictionary(dictionary, negative_emails_sum, 1)
        return pos_dict, neg_dict

    def __get_divergence_results(self, first, second) -> float:
        """
        Obtain Jensen-Shannon divergence based on two input distributions
        """
        log_base = 2  # bits utilized, np.exp(1) for nats
        common = [(p + q) * 0.5 for p, q in zip(first, second)]
        jensen = self.__js_divergence(first, second, common, log_base)
        return jensen

    def __run_with_entity_occurrences(
        self, entity: str, first: dict, second: dict
    ) -> float:
        """
        Based on two named entity occurrences dictionaries, compute
        a Jensen-Shannon divergence for the entity given
        - concrete named entity occurrences are taken into consideration
        """
        v = self.__get_entity_distribution_values(entity, first)
        w = self.__get_entity_distribution_values(entity, second)
        distributions = self.__key_union(v, w)
        first_distrib, second_distrib = self.__get_normalized_values(distributions)
        storage_fv = [v for v in first_distrib.values()]
        storage_sv = [v for v in second_distrib.values()]
        return self.__get_divergence_results(storage_fv, storage_sv)

    def __get_merged_entities(self, entity_values) -> dict:
        """
        Return dictionary in a "Bernoulli" format - non-existent vs found at least once
        """
        merged = {}
        merged["0"] = entity_values["0"]
        merged["at_least_one"] = sum(entity_values.values()) - merged["0"]
        return merged

    def __run_entity_in_sentence(self, entity: str, first: dict, second: dict):
        """
        Based on two named entity occurrences dictionaries, compute
        a Jensen-Shannon divergence for the entity given
        - concrete named entity occurrences are NOT taken into consideration
        """
        v = self.__get_entity_distribution_values(entity, first)
        w = self.__get_entity_distribution_values(entity, second)
        merged_v = self.__get_merged_entities(v)
        merged_w = self.__get_merged_entities(w)

        # it should not happen in case large datasets are considered, but for the convenience
        distributions = self.__key_union(merged_v, merged_w)

        pos_dist, neg_dist = self.__get_normalized_values(distributions)
        pos = [v for v in pos_dist.values()]
        neg = [v for v in neg_dist.values()]
        return self.__get_divergence_results(pos, neg)

    def __write_json(self, data: dict, fname: str):
        with open(fname + ".json", "w") as json_file:
            json.dump(data, json_file, indent=4)


class TTest(Statistics):
    """
    Class where a t-test was performed
    Nevertheless, this method eventually ended up dissatisfying
    with the data I was provided

    - discontinued
    """

    def __init__(self):
        super(TTest, self).__init__()

        self.p_value_mapping = {  # Z_{alpha/2}
            0.001: 3.0923,
            0.005: 2.57583,
            0.01: 2.32635,
            0.025: 1.95996,
            0.05: 1.64485,
            0.1: 1.28155,
        }

    def process_input_files(self):
        self.__process_data(self.positive_emails_data)
        self.__process_data(self.negative_emails_data)

    def calculate_statistics(self):
        """
        Runs the t-test on a per-entity basis
        """
        for idx in range(len(self.stats)):
            for jdx in range(idx + 1, len(self.stats)):
                if (
                    self.stats[idx].model == self.stats[jdx].model
                    and self.stats[idx].entity == self.stats[jdx].entity
                ):
                    self.first = self.stats[idx]
                    self.second = self.stats[jdx]
                    self.__calculate_equation()

    def __process_data(self, data):
        for key in data:
            self.__process_per_model(key, data[key])

    def __process_per_model(self, model_id: str, entities: dict):
        sentence_count = entities["sentences"]
        for ent in entities:
            self.__process_per_entity(model_id, ent, entities[ent], sentence_count)

    def __process_per_entity(self, model_type, entity, occurrences, sentence_count):
        if not entity.islower():  # get rid of _sentences
            occurrence_sum = 0
            for value in occurrences:
                occurrence_sum += occurrences[value] * int(value)
            stat_info = StatInfo(model_type, entity, sentence_count, occurrence_sum)
            self.stats.append(stat_info)

    def __write_to_file(self, t_statistics: float):
        """
        Writes each named entity t-test results per-model to a file
        - model:
            - named entity: <> occurrences of the entity out of <> sentences in positive, then negative emails
            - various p-values measured + hypothesis result
        """
        path_to_file = self.file.get()
        with open(path_to_file, "a+") as file:
            title = "Model: {} Entity: {}\n".format(self.first.model, self.first.entity)
            file.write(title)

            occurrences = "[FIRST]: occurrences: {}, sentences: {} [SECOND]: occurrences: {}, sentences: {}\n".format(
                self.first.occurrence_sum,
                self.first.sentences,
                self.second.occurrence_sum,
                self.second.sentences,
            )
            file.write(occurrences)

            for value in self.p_value_mapping:
                results = "p-value: {:.3f}, Z-value = {:.3f}, |T| = {:.3f}".format(
                    value, self.p_value_mapping[value], t_statistics
                )
                file.write(results)
                if self.p_value_mapping[value] < t_statistics:
                    decision = " - Result: H_0 refuted\n"
                else:
                    decision = " - Result: H_0 accepted\n"
                file.write(decision)
            file.write("\n")

    def __calculate_equation(self):
        """
        Numerics behnd t-test
        """

        theta_first = self.first.occurrence_sum / self.first.sentences
        theta_second = self.second.occurrence_sum / self.second.sentences

        numerator = theta_first - theta_second

        statistics_estimate = (
            self.first.occurrence_sum + self.second.occurrence_sum
        ) / (self.first.sentences + self.second.sentences)

        # the root of variance
        denominator = (
            statistics_estimate
            * (1 - statistics_estimate)
            * (1 / self.first.sentences + 1 / self.second.sentences)
        )

        t_statistics = abs(numerator / math.sqrt(denominator))
        self.__write_to_file(t_statistics)
