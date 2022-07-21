from abc import ABC, abstractmethod
import argparse


class Parser(ABC):
    """
    Common ground for parsers utilized for experiments
    mentioned in this project
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_options()
        self.outdir = "outdir"

    @abstractmethod
    def set_options(self):
        pass

    def get_arguments(self):
        """
        Parse obtained arguments via argparse libraries while
        checking whether the user is not using arguments
        to unintentionally remove files
        - i.e. "--outdir=."
        - no filesystem traversal supported
        """

        parsed = self.parser.parse_args()
        if self.outdir in parsed:
            dictionary = vars(parsed)
            if "." in dictionary[self.outdir]:
                self.__throw_outdir_invalid()
        return parsed

    def __throw_outdir_invalid(self):
        raise ValueError(
            "Output directory shall not contain a dot - Filesystem traversal is not supported."
        )


class DatasetDependentParser(Parser):
    """
    Parsers relying on the Enron email dataset
    """

    def __init__(self):
        super().__init__()

    def set_options(self):
        super().set_options()

        # Suppose the input dataset does not change as soon it is either run via enron_download.py or downloaded
        # from https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
        # - Note that they are interchangeable (nevertheless, be by default Kaggle dataset names their Enron email dataset "emails.csv")
        # Alternatively, other dataset can be utilized
        # - However, it needs to be compliant with the format shown in Kaggle Enron Email Dataset
        self.parser.add_argument("--filename", default="enron.csv", type=str)

        # Various modes are available
        # - full - all libraries included, be vary of its long model loading and computation
        # - NER library group based: spacy, huggingface, flair
        # - concrete models: --mode=sm -> en_core_web_sm (the rest md,lg,trf analogically)
        #                    --mode=hb - dslim-NER-base, hl large analogically
        #                    --mode=fltrf - flair large - --mode=flstm flair fast
        self.parser.add_argument("--mode", default="md", type=str)


class NERParser(DatasetDependentParser):
    """
    Enron experiment parser
    """

    def __init__(self):
        super().__init__()

    def set_options(self):
        super().set_options()

        # Number of emails which are supposed to be processed, be vary that
        # be 1000+ emails starts being computationally demanding on a regular hardware
        self.parser.add_argument("--email_count", default=100, type=int)

        # Output directory
        self.parser.add_argument("--outdir", default="out", type=str)


class AnnotationParser(DatasetDependentParser):
    """
    Annotation experiment parser
    """

    def __init__(self):
        super().__init__()

    def set_options(self):
        super().set_options()

        # Output directory
        self.parser.add_argument("--outdir", default="sents", type=str)


class BenchmarkParser(Parser):
    """
    Benchmark experiment parser
    """

    def __init__(self):
        super().__init__()

    def set_options(self):
        super().set_options()

        # Number of rows (sentences) defaulted to 10000 sentences since the input files do not contain
        # any more lines but a lower number is definitely available for a minor benchmark
        self.parser.add_argument("--sentences", default=10000, type=int)

        # model type - used in the benchmarking experiment
        #            - at this moment, it was already decided that only OntoNotes-based models are supported,
        #            nevertheless, benchmarking can be made on any currently supported model
        #            --mode=md
        self.parser.add_argument("--mode", default="md", type=str)

        self.parser.add_argument("--dir", default="data", type=str)

        # Dataset to form/to use precomputed by the author - for benchmark,
        # - to replicate the experiment, data/seed{0,16,32}_10k.csv on which the measurements were done are prepared in advance
        self.parser.add_argument("--filename", default="seed0_10k", type=str)


class StatisticsExperimentParser(Parser):
    """
    Statistics common ground parsers
    """

    def __init__(self):
        super().__init__()

    def set_options(self):
        self.parser.add_argument("--outdir", default="stats_out", type=str)


class DivergenceParser(StatisticsExperimentParser):
    """
    Divergence experiment parser
    """

    def __init__(self):
        super().__init__()

    def set_options(self):
        super().set_options()
        self.parser.add_argument("--occurrences", action="store_true")
