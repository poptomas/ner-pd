from ner_pd.parsers import StatisticsExperimentParser
from ner_pd.visualizers import ROCVisualizer
from ner_pd.utils import FilesystemUtils, FormatUtils


def main(args):
    FilesystemUtils.clean_previous_run(args.outdir)
    fname = "data/roc_export_ner_results.json"
    v = ROCVisualizer(args)
    v.visualize_roc_curve_plots(fname)


if __name__ == "__main__":
    sent = StatisticsExperimentParser()
    args = sent.get_arguments()
    FormatUtils.print_arguments(args)
    main(args)
