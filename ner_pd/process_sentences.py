from tqdm import tqdm
import pandas as pd

import os
from abc import abstractmethod

from .dataset_processors import SentencesProcessor
from . import ner
from .ner import NER
from gold.evaluate import GoldEval
from .filename import Filename
from .utils import FormatUtils, benchmark


class SentenceRunner:
    """
    SentenceRunner interface serves as a common ground
    for experiments which work on a sentence-level
    - Annotation, benchmark experiment
    """

    @abstractmethod
    def run(self):
        pass

    def prepare(self):
        """
        Opens the dataset provided such dataset exists,
        otherwise an exception is thrown
        """
        try:
            self.__open_dataset()
        except FileNotFoundError as exc:
            print(exc)

    def __open_dataset(self):
        self.dataset = pd.read_csv(self.dataset_fname.get(), header=None)

    @benchmark
    def process_sentences(self, models: list[ner.NER], limit: int, with_results: bool):
        self.dataset = self.dataset[0:limit]
        """
        Runs named entity recognition task on each sentence 
        from the prepared "sentence annotation dataset"
        """
        for _, row in tqdm(
            self.dataset.itertuples(),
            total=len(self.dataset.index),
            colour="red",
            desc="Processing sentences...",
        ):
            for model in models:
                model.run_on_sentence(sentence=row)
                entities = model.get_entities()
                if with_results:
                    self.__write_output(
                        entities=entities, identifier=model.get_identifier()
                    )

    def __write_output(self, entities: list, identifier: str):
        fname = Filename(
            relpath=self.args.outdir, name=identifier, extension=self.models_output_ext
        )
        with open(fname.get(), "a+") as file:
            for idx, entity in enumerate(entities):
                category = entity.type
                ent = entity.text
                file.write("{}:{}".format(category, ent))
                if idx != len(entities) - 1:
                    file.write(";")
            file.write("\n")


class AnnotationRunner(SentenceRunner):
    """
    Annotation experiment runner
    - utilizes handmade annotations on (currently) 200 annotated sentences from the Enron email dataset
    - spaCy & flair models' predictive performance using various evaluation metrics are compared
    """

    def __init__(self, args):
        self.dataset_fname = Filename(
            name="sentences", extension=".csv", relpath=args.outdir
        )
        self.args = args
        self.models_output_ext = ".txt"
        self.limit = 200  # currently annotated
        self.models = []

    def prepare(self):
        self.__prepare_dataset()
        super().prepare()

    def add(self, model: NER):
        self.models.append(model)

    def run(self):
        self.process_sentences(self.models, limit=self.limit, with_results=True)

    def evaluate_findings(self):
        """
        Findings made by models provided are compared to the annotation made
        by the author
        """
        filenames = []
        outdir = self.args.outdir
        for fname in os.listdir(outdir):
            if fname.endswith(self.models_output_ext):
                fname_obj = Filename(
                    relpath=outdir,
                    name=fname[: fname.find(".")],
                    extension=self.models_output_ext,
                )
                filenames.append(fname_obj)
        gold = GoldEval(outdir)
        gold.evaluate(filenames)

    def __prepare_dataset(self):
        sent_proc = SentencesProcessor(args=self.args, outfile=self.dataset_fname.get())
        sent_proc.build_dataset(num_records=self.limit)


class BenchmarkRunner(SentenceRunner):
    """
    Benchmark experiment runner
    - measures the speed of NER models on 10 000 sentences prepared beforehand by the author
    contained in the "data" directory
    """

    def __init__(self, args):
        self.args = args
        self.dataset_fname = Filename(
            name=args.filename, extension=".csv", relpath=args.dir
        )

    def run(self):
        """
        According to the amount of sentences provided by argparse,
        a chosen model's runtime performance is measured
        """
        model_map = NER.get_model_map()
        if self.args.mode in model_map:
            member = model_map[self.args.mode]
            model = member.model_type(member.identifier)
            self.process_sentences(
                [model], with_results=False, limit=self.args.sentences
            )
        else:
            raise ValueError("choose from {}".format([v for v in model_map.keys()]))
