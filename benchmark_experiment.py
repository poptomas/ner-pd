import argparse

from ner_pd.utils import FormatUtils
from ner_pd.process_sentences import BenchmarkRunner
from ner_pd.parsers import BenchmarkParser


def main(args: argparse.Namespace):
    runner = BenchmarkRunner(args)
    runner.prepare()
    runner.run()


if __name__ == "__main__":
    parser = BenchmarkParser()
    args = parser.get_arguments()
    FormatUtils.print_arguments(args)
    main(args)
