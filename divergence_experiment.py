import argparse

from ner_pd.parsers import DivergenceParser
from ner_pd.visualizers import StatisticsExperimentVisualizer
from ner_pd.utils import FormatUtils, FilesystemUtils


def main(args: argparse.Namespace):
    FilesystemUtils.clean_previous_run(args.outdir)
    vis = StatisticsExperimentVisualizer(args)
    vis.visualize_probabilities()
    vis.visualize_divergence_alltogether()
    format_ = "pdf"
    FilesystemUtils.clean_up_besides(format_, args.outdir)


if __name__ == "__main__":
    parser = DivergenceParser()
    args = parser.get_arguments()
    FormatUtils.print_arguments(args)
    main(args)
