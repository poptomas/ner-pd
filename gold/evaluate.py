import os

from ner_pd.ner import NamedEntity
from ner_pd.filename import Filename
from ner_pd.utils import DatasetUtils, DictionaryUtils, StringUtils

import pandas as pd


class GoldEval:
    """
    A class that serves as an evaluator
    to compare the predictive performance
    of various models based on the prior manual
    annotation considered "gold" data
    """

    def __init__(self, outdir: str):
        self.gold_fname = Filename(relpath="gold", name="sentences", extension=".gold")

        self.format = ".csv"
        self.outdir = outdir
        self.table_fname = "table"
        self.stats_fname = "score"

        self.true_pos = "True Positive"
        self.false_pos = "False Positive"
        self.false_neg = "False Negative"
        self.almost_tp = "Almost True Positive"
        self.col_names = [self.true_pos, self.false_pos, self.false_neg, self.almost_tp]
        self.data = {}
        self.stats = {}

        self.name_col = "NAME"
        self.ent_col = "ENTITY"

    def evaluate(self, filenames: list[Filename]):
        """
        Processes data requested for the evaluation
        Afterward, each model requested for analysis
        runs NER on prepared sentences, and is compared
        against the author's gold data annotations
        """
        self.__process_data(filenames)
        self.__analyze_all()

    def __process_data(self, filenames: list):
        """
        Process lines of gold data and process each
        filename (resp. model representation) obtained for the evaluation
        """
        with open(self.gold_fname.get(), "r") as gold:
            gold_data = gold.readlines()
            self.__process_lines(gold_data, self.gold_fname)
            for file in filenames:
                self.__process_file(file)

    def __process_file(self, fname: Filename):
        with open(fname.get(), "r") as file:
            reader = file.readlines()
            self.__process_lines(reader, fname)

    def __process_lines(self, data: list[str], fname: Filename):
        """
        Form named entities from the obtained raw data
        """
        split_char = ";"
        for line in data:
            line = line.strip()
            split_line = line.split(split_char)
            self.__form_named_entities(split_line, fname)

    def __form_named_entities(self, entity_list: list[str], fname: Filename):
        """
        Form named entities in NamedEntity dataclass format - NamedEntity(type: str, text: str)
        """
        named_entities = []
        for entity_pair in entity_list:
            pair = entity_pair.split(":")
            if len(pair) == 2:  # last endline stays alone
                named_entities.append(NamedEntity(pair[0], pair[1]))
        name = fname.get_name()
        if name not in self.data:
            self.data[name] = [named_entities]
        else:
            self.data[name].append(named_entities)

    def __analyze_all(self):
        """
        Each model named entities found are compared
        against the gold data annotated by the author
        """
        gold = self.data[self.gold_fname.get_name()]
        for identifier, findings in self.data.items():
            if identifier != self.gold_fname.get_name():
                self.__analyze_model(findings, gold, identifier)
        self.__write_stats()

    def __analyze_model(
        self,
        found_entities: list[list[NamedEntity]],
        gold_entities: list[list[NamedEntity]],
        identifier: str,
    ):
        """
        Model's found named entities are compared with
        the author's gold named entity annotation
        - various evaluation metrics are calculated
        - the results are written to a CSV file
        """
        self.distrib_dict = {}
        for line, gold_line in zip(found_entities, gold_entities):
            self.__analyze_line(line, gold_line, identifier)
        initial_cols = [self.name_col]

        for entity in self.distrib_dict.keys():
            dataframe = pd.DataFrame(columns=initial_cols)
            for col in self.col_names:
                dataframe[col] = pd.Series(dtype=int)
            stats = {}
            for indicator in self.distrib_dict[entity]:
                stats[indicator.outcome] = indicator.count
            stats[self.name_col] = identifier

            dataframe.loc[len(dataframe.index)] = stats

            dataframe = self.__calculate_score(dataframe)
            output = Filename(relpath=self.outdir, name=entity, extension=self.format)
            fname = output.get()
            if os.path.exists(fname):
                dataframe.to_csv(fname, index=False, mode="a", header=False)
            else:
                dataframe.to_csv(fname, index=False, mode="w", header=True)

    def __analyze_entity(
        self, entity: NamedEntity, gold_entities: list[NamedEntity], identifier: str
    ) -> bool:
        """
        Processes a named entity and assigns it
        to a dictionary based on the correction of the verdict
        - false positive x true positive, resp. almost true positive for the relaxed F1-score
        """
        found_index = -1
        for index, gold in enumerate(gold_entities):
            if entity.text == gold.text and entity.type == gold.type:
                found_index = index
                self.found_ents.append(entity)
                DictionaryUtils.increment_nested_occurrence(
                    self.distrib_dict, entity.type, self.true_pos
                )
                DictionaryUtils.increment_nested_occurrence(
                    self.stats, identifier, self.true_pos
                )
                break
            elif (entity.text == gold.text and entity.type != gold.type) or (
                entity.type == gold.type
                and (entity.text in gold.text or gold.text in entity.text)
            ):  # both ways substring with a properly inferred entity type
                found_index = index
                self.almost_found_ents.append(entity)
                DictionaryUtils.increment_nested_occurrence(
                    self.distrib_dict, entity.type, self.almost_tp
                )
                DictionaryUtils.increment_nested_occurrence(
                    self.stats, identifier, self.almost_tp
                )
                break
        if found_index != -1:
            self.gold_indices[found_index] = True
        else:
            self.false_positives.append(entity)
            DictionaryUtils.increment_nested_occurrence(
                self.distrib_dict, entity.type, self.false_pos
            )
            DictionaryUtils.increment_nested_occurrence(
                self.stats, identifier, self.false_pos
            )

    def __analyze_line(
        self,
        found_entities: list[NamedEntity],
        gold_entities: list[NamedEntity],
        identifier: str,
    ):
        """
        Launches per-sentence categorization of
        found named entities by the model
        and named entities annotated by the author
        - true positive, false positive, false negative
        """
        # adjustment to avoid way too long lines which are a signal of a suspicious sentence
        # for instance, a table or multiple times forwarded emails
        # (finds plenty of person entities without much effort)
        # - also annotations were not made for such outlier sentences
        # - probably the number could be even lower
        max_entities_per_sentence = 20
        self.almost_found_ents = []
        self.found_ents = []
        self.false_positives = []
        self.false_negatives = []
        # to find false negatives without multiple gold data traversal
        self.gold_indices = [False for _ in gold_entities]
        if len(found_entities) < max_entities_per_sentence:
            for entity in found_entities:
                self.__analyze_entity(entity, gold_entities, identifier)
            for was_found, gold in zip(self.gold_indices, gold_entities):
                if not was_found:
                    self.false_negatives.append(gold)
                    DictionaryUtils.increment_nested_occurrence(
                        self.distrib_dict, gold.type, self.false_neg
                    )
                    DictionaryUtils.increment_nested_occurrence(
                        self.stats, identifier, self.false_neg
                    )
        self.__write_results(identifier)

    def __calculate_score(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates precision, recall, F1-score, and its relaxed alternatives
        """

        round_dec_places = 3

        precision = dataframe[self.true_pos] / (
            dataframe[self.true_pos] + dataframe[self.false_pos]
        )
        recall = dataframe[self.true_pos] / (
            dataframe[self.true_pos] + dataframe[self.false_neg]
        )

        # precision, recall including almost true positives
        precision_atp = (dataframe[self.true_pos] + dataframe[self.almost_tp]) / (
            dataframe[self.true_pos]
            + dataframe[self.almost_tp]
            + dataframe[self.false_pos]
        )  #
        recall_atp = (dataframe[self.true_pos] + dataframe[self.almost_tp]) / (
            dataframe[self.true_pos]
            + dataframe[self.almost_tp]
            + dataframe[self.false_neg]
        )  #

        dataframe["precision"] = round(precision, round_dec_places)
        dataframe["recall"] = round(recall, round_dec_places)
        dataframe["F1-score"] = round(
            2 * precision * recall / (precision + recall), round_dec_places
        )
        relaxed_f1 = "F1-score with {}".format(self.almost_tp)
        dataframe[relaxed_f1] = round(
            2 * precision_atp * recall_atp / (precision_atp + recall_atp),
            round_dec_places,
        )

        dataframe = dataframe.fillna(0)
        return dataframe

    def __write_stats(self):
        """
        Stats CSV file is created based
        on the accumulated stats dictionary
        - models are compared using various evaluation metrics as follows:
        NAME,True Positive,False Positive,False Negative,Almost True Positive,precision,recall,F1-score,F1-score with Almost True Positive
        en_core_web_lg,...
        en_core_web_sm,....
        en_core_web_md,...
        en_core_web_trf,...
        - produced using python annotation_experiment.py --mode=spacy
        """

        output_fname = Filename(
            relpath=self.outdir, name=self.stats_fname, extension=self.format
        )
        dataframe = pd.DataFrame(columns=[self.name_col])
        for col in self.col_names:
            dataframe[col] = pd.Series(dtype=int)

        for identifier in self.stats.keys():
            if identifier != self.gold_fname.get_name():
                stats = {}
                for stat in self.stats[identifier]:
                    stats[stat.outcome] = stat.count
                stats[self.name_col] = identifier
                dataframe.loc[len(dataframe.index)] = stats
        dataframe = self.__calculate_score(dataframe)
        output_fname = Filename(
            relpath=self.outdir, name=self.stats_fname, extension=self.format
        )
        dataframe.to_csv(output_fname.get(), index=False)

    def __write_results(self, identifier):
        output_fname = Filename(
            relpath=self.outdir,
            name="{}_{}".format(identifier, self.table_fname),
            extension=self.format,
        )
        row_records = [
            StringUtils.format_found_ents(self.found_ents),
            StringUtils.format_found_ents(self.false_positives),
            StringUtils.format_found_ents(self.false_negatives),
            StringUtils.format_found_ents(self.almost_found_ents),
        ]
        if os.path.exists(output_fname.get()):
            DatasetUtils.append_row_to_csv(
                table_fname=output_fname.get(),
                row_records=[row_records],
                col_names=[self.col_names],
            )
        else:
            DatasetUtils.write_row_to_csv(
                table_fname=output_fname.get(),
                row_records=[row_records],
                col_names=[self.col_names],
                header=True,
            )
