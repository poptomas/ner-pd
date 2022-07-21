from abc import abstractmethod, ABC
import argparse
from dataclasses import dataclass
import collections
import shutil
import os
import re
import json
from typing import Tuple

import transformers
import spacy

import flair
from flair.data import Sentence
from flair.models import SequenceTagger

import pandas as pd
from tqdm import tqdm

from .filename import Filename
from .dataset_processors import EmailProcessor
from .sentencizer import Sentencizer
from .utils import DictionaryUtils, DatasetUtils, FormatUtils, StringUtils, benchmark

# in order not to spam the console
# - for flair, dlsim(huggingface) models
import warnings

warnings.filterwarnings("ignore")


@dataclass
class NamedEntity:
    type: str
    text: str


class NER(ABC):
    """
    Named entity recognition interface
    - contains the contract of methods which are need to be
    supported to fulfill created requirements
    - run on a piece of text - email included
    - run on a sentence
    - load the model
    - validate proper use of the class by checking its identifier
    - save obtained results
    """

    def __init__(self, identifier: str):
        self.identifier = identifier
        self.model = None
        self.entities = []
        self.stats = {}
        self.entity_dictionary = DictionaryUtils.DictionaryWrapper()
        self.sentence_count = 0
        FormatUtils.print_identifier(self.identifier)

    @abstractmethod
    def run_on_text(self, text: str):
        pass

    @abstractmethod
    def run_on_sentence(self, sentence: str):
        pass

    @abstractmethod
    def load_model(self, model_name: str):
        pass

    @abstractmethod
    def is_identifier_valid(self):
        pass

    @abstractmethod
    def save_results(self):
        pass

    def set_entity_occurences(self):
        for entity in self.entities:
            key = entity.type
            DictionaryUtils.increment_occurrence(self.stats, key)

    def get_stats(self):
        return self.stats

    def set_entities(self, in_list: list):
        self.entities = in_list

    def get_entities(self) -> list[NamedEntity]:
        return self.entities

    def get_sentence_count(self) -> int:
        return self.sentence_count

    def get_identifier(self) -> str:
        # ensure directory issues do not occur due to the name (i.e. dslim/bert-base-NER)
        return self.identifier.replace("/", "-")

    @staticmethod
    def __throw_invalid_identifier(identifier: str):
        raise Exception("Invalid identifier: {}".format(identifier))

    @staticmethod
    def get_model_map() -> dict:
        model_map = {
            "sm": ModelInfo("en_core_web_sm", SpacyTransitionModel),
            "md": ModelInfo("en_core_web_md", SpacyTransitionModel),
            "lg": ModelInfo("en_core_web_lg", SpacyTransitionModel),
            "trf": ModelInfo("en_core_web_trf", SpacyTransformer),
            "hbase": ModelInfo("dslim/bert-base-NER", DslimNER),
            "hlarge": ModelInfo("dslim/bert-large-NER", DslimNER),
            "fllstmdefault": ModelInfo("flair/ner-english-ontonotes", BiLSTMModel),
            "fllstm": ModelInfo("flair/ner-english-ontonotes-fast", BiLSTMModel),
            "fltrf": ModelInfo("flair/ner-english-ontonotes-large", FlairTransformer),
        }
        return model_map

    @staticmethod
    def get_group_map() -> dict:
        group_map = {
            "spacy": "en_core_web_",
            "huggingface": "dslim/bert-",
            "flair": "flair/ner-english",
        }
        return group_map


@dataclass
class ModelInfo:
    identifier: str
    model_type: type[NER]


class HuggingFaceNER(NER):
    """
    Implements interface methods which
    are supposed to be common for various
    models from Hugging Face
    - run on a piece of text - utilizing the Sentencizer,
    then on a sentence level model(input_sentence: str) produces a dictionary
    of "service info" from which the named entities can be taken from
    - run on a sentence
    - model loading - for NER task should remain common for the models
    """

    def __init__(self, identifier: str):
        super(HuggingFaceNER, self).__init__(identifier)
        self.load_model()

    def run_on_text(self, text: str):
        """
        Utilizes the Sentencizer (by spaCy) to parse the input text
        (here email) into sentences
        then proceeds to run per-sentence
        """
        sentencizer = Sentencizer(text)
        self.entities = []
        sentences = sentencizer.get_sentences()
        for sentence in sentences:
            results = self.model(sentence)
            self.save_results(results)
        self.sentence_count += len(sentences)

    def run_on_sentence(self, sentence: str):
        """
        Runs the model on an input sentence
        model(input_sentence: str) produces
        a list of dictionaries containing "service" data
        from which results can be stored
        """
        self.entities = []
        results = self.model(sentence)
        self.save_results(results)
        self.sentence_count += 1

    @abstractmethod
    def is_identifier_valid(self):
        pass

    @abstractmethod
    def save_results(self, results: list):
        pass

    @benchmark
    def load_model(self):
        """
        Utilizes transformers pipeline for pretrained models from HF
        suited for named entity recognition
        - parameters used
            - tokenizer - https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer
            - model - https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForTokenClassification
            - aggregation_strategy="max" - aggregation strategy while classifying the tokens
                - various alternatives: https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/token_classification.py

        """
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.identifier)  #
        model = transformers.AutoModelForTokenClassification.from_pretrained(
            self.identifier
        )  #
        self.model = transformers.pipeline(
            task="ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="max",
            device=0,  # GPU
        )


class DslimNER(HuggingFaceNER):
    """
    'Sealed' class of the NER hierarchy
    - completes the interface by validating the proper use of identifiers
    - model-specific results handling
    - dslim/bert-base-NER - https://huggingface.co/dslim/bert-base-NER
    - dslim/large-NER - https://huggingface.co/dslim/bert-large-NER
    """

    def __init__(self, identifier: str):
        super(DslimNER, self).__init__(identifier)
        if not self.is_identifier_valid():
            NER.__throw_invalid_identifier(identifier)

    def save_results(self, results: list[dict]):
        """
        Model-specific - produces a list of dictionaries from which the information about
        the entities are obtained
        - for instance, [{'entity_group': 'PER', 'score': 0.9976821, 'word': 'Jeff', 'start': 0, 'end': 4}, {...}, ...]
        - named entities are collected as NamedEntity dataclasses
        - accumulates named entity occurrences per sentence for each named entity
        type separately
        """
        dictionary = {}
        for res in results:
            entity_label = res["entity_group"]
            entity = res["word"]
            kvpair = NamedEntity(type=entity_label, text=entity)
            self.entities.append(kvpair)
            DictionaryUtils.increment_occurrence(dictionary, entity_label)
        for key, value in dictionary.items():
            DictionaryUtils.increment_occurrence(self.entity_dictionary[key], value)

    def is_identifier_valid(self):
        """
        Check whether model chosen is compliant with allowed model type
        supported by the class
        """
        name = "dslim/bert-{}-NER"
        return self.identifier == name.format("base") or self.identifier == name.format(
            "large"
        )


class FlairNER(NER):
    """
    Implements interface methods which
    are supposed to be common for Flair model variants
    - run on a piece of text
        - utilizing the Sentencizer + sentence-level processing via flair.data.Sentence(input_sentence: str)
    - run on a sentence - flair.data.Sentence(input_sentence: str)
        - common for the models
    - model loading - flair-specific, common for the models
    """

    def __init__(self, identifier: str):
        self.prefix = "flair/ner-english"
        super(FlairNER, self).__init__(identifier)
        if not self.is_identifier_valid():
            NER.__throw_invalid_identifier(identifier)

    def run_on_text(self, text: str):
        """
        - using Sentencizer (by spaCy),
        the input text is split into sentences
        - afterward, flair represents the sentence
        by its own flair.data.Sentence(str), and
        predicts named entities found on a sentence level
        """
        self.entities = []
        sent = Sentencizer(text)
        sentences = sent.get_sentences()
        for sent in sentences:
            # flair needs own representation of the sentence
            flair_sentence = Sentence(sent)
            self.model.predict(flair_sentence)
            self.save_results(flair_sentence)
        self.sentence_count += len(sentences)

    def run_on_sentence(self, sentence: str):
        """
        - flair represents the sentence
        by its own flair.data.Sentence(str)
        - afterward, NER on a sentence level is performed
        """
        self.entities = []
        flair_sent = Sentence(sentence)
        self.model.predict(flair_sent)
        self.save_results(flair_sent)
        self.sentence_count += 1

    def save_results(self, results: list[flair.data.Sentence]):
        """
        dictionaries which contain named entity occurrences are incremented
        according to named entity types found
        - named entities are collected as NamedEntity dataclasses
        - accumulates named entity occurrences per sentence for each named entity
        type separately
        """
        dictionary = {}
        for entity in results.get_spans("ner"):
            entity_type = entity.get_label("ner").value
            if entity_type == "O":  # not an entity
                continue
            kvpair = NamedEntity(type=entity_type, text=entity.text)
            self.entities.append(kvpair)
            DictionaryUtils.increment_occurrence(dictionary, entity_type)
        for key, value in dictionary.items():
            DictionaryUtils.increment_occurrence(self.entity_dictionary[key], value)

    @benchmark
    def load_model(self):
        self.model = SequenceTagger.load(self.identifier)

    @abstractmethod
    def is_identifier_valid(self):
        pass


class BiLSTMModel(FlairNER):
    """
    'Sealed' class of the NER hierarchy
    - completes the interface by validating the proper use of identifiers
    - ner-ontonotes-fast
        - fine-tuned on CoNLL 2003
        - 1024 hidden states, runs on CPU
        - https://huggingface.co/flair/ner-english-fast
        - support needs to be manually added
            - not supported by the model map
    - ner-english-ontonotes-fast
        - fine-tuned on OntoNotes v5
        - 1024 hidden states, runs on CPU
        - https://huggingface.co/flair/ner-english-ontonotes-fast
    - ner-english
        - unused in the thesis due to slow runtime and poor predictive performance
        - 2048 hidden states, runs on GPU
        - support needs to be manually added
            - not supported by the model map
    - ner-english-ontonotes
        - OntoNotes v5
        - https://huggingface.co/flair/ner-english-ontonotes
    """

    def __init__(self, identifier: str):
        super(BiLSTMModel, self).__init__(identifier)
        if not self.is_identifier_valid():
            NER.__throw_invalid_identifier(identifier)
        self.load_model()

    def is_identifier_valid(self):
        """
        Check whether model chosen is compliant with allowed model suffixes
        supported by the class
        """
        # a flair/ner-english-fast, a lightweight version of flair BiLSTM
        # was used instead of the "regular one" since it showed
        # still one of the worst results but better than the regular "ner-english-ontonotes"
        allowed_endings = ["-ontonotes", "-fast", "-ontonotes-fast"]
        for ending in allowed_endings:
            if self.prefix + ending == self.identifier:
                return True
        return False


class FlairTransformer(FlairNER):
    """
    'Sealed' class of the NER hierarchy
    - completes the interface by validating the proper use of identifiers
    - models mentioned differ only on the dataset they are fine-tuned

    - ner-english-ontonotes-large
        - fine-tuned on OntoNotes v5
        - RoBERTa large, runs on GPU
        - https://huggingface.co/flair/ner-english-ontonotes-large
    - ner-english-large
        - fine-tuned on CoNLL
    """

    def __init__(self, identifier: str):
        super(FlairTransformer, self).__init__(identifier)
        if not self.is_identifier_valid():
            NER.__throw_invalid_identifier(identifier)
        self.load_model()

    def is_identifier_valid(self):
        """
        Check whether model chosen is compliant with allowed model suffixes
        supported by the class
        """
        allowed_endings = ["-large", "-ontonotes-large"]
        for ending in allowed_endings:
            if self.prefix + ending == self.identifier:
                return True
        return False


class SpacyNER(NER):
    """
    Implements interface methods which
    are supposed to be common for spaCy model variants
    - run on a piece of text - utilizing the Sentencizer + spacy.tokens.doc.Doc.ents
    - run on a sentence - spacy.tokens.doc.Doc.ents
    - model loading - remains identic for the spaCy models
    """

    def __init__(self, identifier: str):
        self.prefix = "en_core_web_"
        self.disabled_components = []
        super(SpacyNER, self).__init__(identifier)

    def run_on_text(self, text: str):
        """
        SpaCy's Sentencizer is utilized
        to split the input text into sentences
        - predictions on a sentence level are performed
        - model(input_sentence: str) produces a spaCy
        document (spacy.tokens.doc.Doc) for various purposes:
            - https://spacy.io/api/doc
            - .ents - a named entity tags are assigned
        """
        self.entities = []
        sent = Sentencizer(text)
        sentences = sent.get_sentences()
        for sent in sentences:
            document = self.model(sent)
            self.save_results(document.ents)
        self.sentence_count += len(sentences)

    def run_on_sentence(self, sentence: str):
        """
        NER on a sentence level is performed
        - model(input_sentence: str) produces a spaCy
        document (spacy.tokens.doc.Doc) for various purposes:
            - https://spacy.io/api/doc
            - .ents - a named entity tags are assigned
        """
        self.entities = []
        document = self.model(sentence)
        self.save_results(document.ents)
        self.sentence_count += 1

    @benchmark
    def load_model(self):
        self.model = spacy.load(self.identifier, disable=self.disabled_components)

    @abstractmethod
    def is_identifier_valid(self):
        pass

    def save_results(self, results: list[spacy.tokens.span.Span]):
        """
        list of spaCy "spans" - slices obtained from the text/document
        - spacy.tokens.span.Span - https://spacy.io/api/span
            - .label_ - required for NER
        - named entities are collected as NamedEntity dataclasses
        - accumulates named entity occurrences per sentence for each named entity
        type separately
        """
        dictionary = {}
        for entity in results:
            kvpair = NamedEntity(type=entity.label_, text=str(entity))
            self.entities.append(kvpair)
            DictionaryUtils.increment_occurrence(dictionary, entity.label_)
        for key, value in dictionary.items():
            DictionaryUtils.increment_occurrence(self.entity_dictionary[key], value)


class SpacyTransitionModel(SpacyNER):
    """
    'Sealed' class of the NER hierarchy
    - completes the interface by validating the proper use of identifiers
    - spaCy transition-based model - https://www.youtube.com/watch?v=sqDHBH9IjRU
    - models mentioned differ in size, the amount of static word vectors the model is given
    - en_core_web_sm, en_core_web_md, en_core_web_lg - https://spacy.io/models
    """

    def __init__(self, identifier: str):
        super(SpacyTransitionModel, self).__init__(identifier)
        if not self.is_identifier_valid():
            NER.__throw_invalid_identifier(identifier)
        self.disabled_components = [  # speedup the computation
            "tagger",
            "parser",
            "attribute_ruler",
            "lemmatizer",
            "tok2vec",
        ]
        self.load_model()

    def is_identifier_valid(self):
        """
        Set allowed suffixes to ensure that
        only models in the user code are supported
        """
        allowed_models = ["sm", "md", "lg"]
        for model in allowed_models:
            if self.prefix + model == self.identifier:
                return True
        return False


class SpacyTransformer(SpacyNER):
    """
    'Sealed' class of the NER hierarchy
    - spaCy transformed-based model
    - RoBERTa-base fine-tuned on OntoNotes v5
    """

    def __init__(self, identifier: str):
        super(SpacyTransformer, self).__init__(identifier)
        if not self.is_identifier_valid():
            NER.__throw_invalid_identifier(identifier)
        self.disabled_components = ["tagger", "parser", "attribute_ruler", "lemmatizer"]
        self.load_model()

    def is_identifier_valid(self):
        """
        Set allowed suffixes to ensure that
        only models in the user code are supported
        """
        allowed_models = "trf"
        return self.prefix + allowed_models == self.identifier


class ModelRunner:
    """
    Enron experiment runner
    - launches chosen NER models as was entered on the commandline
    - generates various analysis-based outputs
        1) CSV pairwise files of comparisons,
        2) JSON produced contains named entity occurrences - used for the KL/JS divergence experiment,
        3) CSV containing per-model count of each named entity type found
    """

    def __init__(self, email_processor: EmailProcessor, args: argparse.Namespace):
        self.args = args
        self.models = []
        self.email_processor = email_processor
        self.output_directory = args.outdir

        self.dataset_extension = ".csv"
        self.stats_outfile = "stats"
        self.common_outfile = "all"
        self.entities_outfile = "entities"
        self.sentence_col = "Original sentence"
        self.comparison_colnames = [
            "Intersection",
            "Difference (A\B)",
            "Difference (B\A)",
            "Sentence",
        ]

    def prepare_directory(self):
        """
        Clean previous run
        """
        if os.path.exists(self.output_directory):
            shutil.rmtree(self.output_directory)
        os.makedirs(self.output_directory)

    def add(self, *models: Tuple[NER]):
        """
        Add models for named entity recognition task
        """
        for m in models:
            self.models.append(m)

    @benchmark
    def process_emails(self, iterations: int):
        """
        Process emails, tqdm progress bar included
        to provide an overview - runtime + current progress
        """
        for _ in tqdm(range(iterations), colour="red"):
            self.__process_email()

    def write_stats(self):
        """
        Write analysis output files containing
        1) CSV pairwise files of comparisons,
        2) JSON produced contains named entity occurrences - used for the KL/JS divergence experiment,
        3) CSV containing per-model count of each named entity type found
        """
        self.__compare_models()
        self.__count_entity_occurrences()
        self.__count_model_entities()

    def __process_email(self):
        """
        Creates an email using EmailProcessor
        - an obtained cleaned email is then passed to NER model(s)
        """
        self.email_processor.create_email()
        text = self.email_processor.get_text()
        text = text.replace("\n", " ")
        self.__run_all_models(text)

    def __run_all_models(self, text: str):
        """
        Each chosen model runs on the input text
        - afterward, results are saved to a table
        """
        for model in self.models:
            self.__run_model(model, text)
        self.__save_results_to_table(text)

    def __run_model(self, model: NER, text: str):
        """
        Run the model on a piece of text
        and report the results of found named entities
        """
        model.run_on_text(text)
        model.set_entity_occurences()

    def __get_entity_set(self, line: str, model: NER) -> set:
        """
        A line (a csv row) contains
        i.e. [PER: Jeff] [PER: John Blaisdell] [ORG: Photofete],[PER: Jeff] [PER: John Blaisdell] [ORG: Photofete]
        where each comma delimits one column where model found named entities reside
        - the aim is to capture the braces
        to end up solely with the contents as a set
        """
        ents = line[model.get_identifier()]
        capture_braces_regex = "\[.*?\]"
        return set(re.findall(capture_braces_regex, ents))

    def __get_comparison_operations(
        self, line: str, first_m: NER, second_m: NER
    ) -> list:
        """
        Perform intersection, (assuming A, B are sets) A-B diff, B-A diff, and
        return the original sentence to form columns for the output comparison csv file
        """
        sentence = line[self.sentence_col]
        first = self.__get_entity_set(line, first_m)
        second = self.__get_entity_set(line, second_m)
        intersection = first.intersection(second)
        a_b_diff = first.difference(second)
        b_a_diff = second.difference(first)
        return [
            StringUtils.get_string_from_set(intersection),
            StringUtils.get_string_from_set(a_b_diff),
            StringUtils.get_string_from_set(b_a_diff),
            sentence,
        ]

    def __write_comparison(
        self, row_records: list, first_model_n: str, second_model_n: str
    ):
        """
        Write obtained comparison operations in the format
        - interection,differnece(A-B),difference(B-A),original sentence
        """
        column_names = self.comparison_colnames
        filename = Filename(
            relpath=self.output_directory,
            name="comparison-{}-{}".format(first_model_n, second_model_n),
            extension=self.dataset_extension,
        )

        DatasetUtils.write_row_to_csv(
            table_fname=filename.get(),
            row_records=row_records,
            col_names=column_names,
            header=True,
        )

    def __compare_models(self):
        """
        A simple comparison used for per-library differences in named entities found
        - comparison is launched only for the "group" or "full" mode
        - it was mainly used for spacy models to
        find anomalies in named entity differences found
        between transformer and transition-based models
        """
        if len(self.models) < 2:
            return
        common_count = 5
        for i in range(len(self.models)):
            for j in range(i + 1, len(self.models)):
                first_idf = self.models[i].get_identifier()
                second_idf = self.models[j].get_identifier()
                if StringUtils.has_same_prefix(first_idf, second_idf, common_count):
                    row_records = []

                    filename = Filename(
                        relpath=self.output_directory,
                        name="all",
                        extension=self.dataset_extension,
                    )

                    dataframe = pd.read_csv(filename.get())
                    for _, line in dataframe.iterrows():
                        row = self.__get_comparison_operations(
                            line, self.models[i], self.models[j]
                        )
                        row_records.append(row)
                    self.__write_comparison(row_records, first_idf, second_idf)

    def __count_model_entities(self):
        """
        Form a csv table containig named entity occurrences,
        an example of spacy transformer running on 100k emails from Enron email dataset
        - python ner.py --email_count=100000 --mode=trf --outdir=100k --with_analysis
        NAME,ALL,PERSON,DATE,ORG,ORDINAL,TIME,PRODUCT,MONEY,GPE,PERCENT,CARDINAL,LAW,WORK_OF_ART,LOC,NORP,EVENT,FAC,QUANTITY,LANGUAGE
        en_core_web_trf,2488077,1083803,281949,334837,21291,234157,25372,39435,214491,9845,156733,4967,5761,16373,25404,3294,19673,9875,817
        """

        name_col = "NAME"
        sum_col = "ALL"
        dataframe = pd.DataFrame(columns=[name_col, sum_col])

        filename = Filename(
            relpath=self.output_directory,
            name=self.stats_outfile,
            extension=self.dataset_extension,
        )

        for model in self.models:
            for s in model.get_stats():
                if s not in dataframe:
                    dataframe[s] = pd.Series(dtype=int)

        for model in self.models:
            identifier = model.get_identifier()
            stats = model.get_stats()
            stats[sum_col] = sum(stats.values())
            stats[name_col] = identifier
            new_row = pd.DataFrame([stats])
            dataframe = pd.concat(
                [dataframe, new_row], axis=0, join="outer", ignore_index=True
            )
        dataframe = dataframe.fillna(0)
        # Get rid of decimal numbers due to zero filling
        dataframe = pd.concat(
            [dataframe[name_col], dataframe.iloc[:, 1:].astype(int)], axis=1
        )
        dataframe.to_csv(filename.get(), index=False)

    def __count_entity_occurrences(self):
        final = {}
        for model in self.models:
            idf = model.get_identifier()
            for key in model.entity_dictionary:
                model.entity_dictionary[key] = collections.OrderedDict(
                    sorted(model.entity_dictionary[key].items())
                )
            model.entity_dictionary["sentences"] = model.get_sentence_count()
            final[idf] = model.entity_dictionary
        fname = Filename(
            relpath=self.output_directory, name=self.entities_outfile, extension=".json"
        )
        with open(fname.get(), "w") as file:
            json.dump(final, file, indent=2)

    def __process_entities(
        self, sentence: str, entities: list[NamedEntity]
    ) -> Tuple[collections.deque[NamedEntity], list[NamedEntity]]:
        """
        Run across named entities found (on an email level)
        on a sentence (sequentially), so
        gradually reduce the named entities by encountering sentences of the email
        - deque is used exactly for this reason
        - a slight optimization was made since in case the entity is found,
        from this part, there is no need to keep searching from the very start
        of the sentence
        """
        entity_deque = collections.deque(entities)
        copy_ents = entity_deque.copy()
        found_ents = []
        for ent in entity_deque:
            entity = ent.text
            prev_sentence = sentence
            sentence = StringUtils.trim_front_if_contained(sentence, entity)
            if sentence != prev_sentence:
                found_ents.append(ent)
                copy_ents.popleft()
            else:
                break
        return copy_ents, found_ents

    def __save_results_to_table(self, text: str):
        """
        Write results to a csv table
        by processing assigning named entities
        to the sentences where they were found in an email
        - csv example:
        model1,model2,...,Original sentence
        [PERSON: Jeff],[PERSON:Jeff][PERSON: John Blaisdell],...,"Jeff, John Blaisdell..."
        """
        sent = Sentencizer(text)
        sentences = sent.get_sentences()

        copy = self.models.copy()
        row_records = []
        for sent in sentences:
            row = []
            for model in copy:
                mod_entities = model.get_entities()
                mod_entities, found_entities = self.__process_entities(
                    sent, mod_entities
                )
                model.set_entities(mod_entities)
                result = StringUtils.format_found_ents(found_entities)
                row.append(result)
            row.append(sent)
            row_records.append(row)
        column_names = [m.get_identifier() for m in self.models]
        column_names.append(self.sentence_col)

        fname = Filename(
            relpath=self.output_directory,
            name=self.common_outfile,
            extension=self.dataset_extension,
        )
        all_fname = fname.get()

        if os.path.exists(all_fname):
            DatasetUtils.append_row_to_csv(
                table_fname=all_fname, row_records=row_records, col_names=column_names
            )
        else:
            DatasetUtils.write_row_to_csv(
                table_fname=all_fname,
                row_records=row_records,
                col_names=column_names,
                header=True,
            )
