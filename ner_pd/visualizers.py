import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from .statistics import Divergence

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Visualizer header with the main, straightforward purpose
to obtain desired plots included in the thesis
- mainly to fulfill the purpose
- matplotlib is utilized
"""


class ROCVisualizer:
    """
    Class utilized for displaying ROC curves
    from provided JSON file located in "data" directory
    with a filename visibly distinguished from the rest
    """

    def __init__(self, args: argparse.Namespace):
        self._args = args

    def visualize_roc_curve_plots(self, fname: str):
        """
        Create plots of monthly ROC curves from March and April
        Afterward, time-based train-test split ROC curve is visualized as a plot
        """

        # JSON export-specific
        false_positive_rate_key = "FPRs"
        true_positive_rate_key = "TPRs"
        train_test_plot_key = "TrainTestSplit"

        with open(fname, "r") as file:
            contents = json.load(file)
            plots = pd.DataFrame.from_dict(contents)
            for plot_name in plots.columns:
                plot_type = plots[plot_name]
                keys = plot_type.keys()
                if plot_name == train_test_plot_key:
                    for key, value in plot_type.items():
                        false_pos_rates = value[false_positive_rate_key]
                        true_pos_rates = value[true_positive_rate_key]
                        plt.plot(false_pos_rates, true_pos_rates)
                        ## To obtain the values at various thresholds
                        # points = [0.001, 0.01, 0.1]
                        # self.__interp_points(false_pos_rates, true_pos_rates, points)
                    self.__plot_train_test_roc(plot_name)
                else:
                    for model, model_results in plot_type.items():
                        val = 0
                        fpr_round_kfolds, tpr_round_kfolds = self.__get_folds(
                            model_results,
                            false_positive_rate_key,
                            true_positive_rate_key,
                        )
                        self.__interpolate(fpr_round_kfolds, tpr_round_kfolds)
                    self.__plot_per_month_roc(plot_name)

    def __get_folds(self, model_results: list[list[dict]], fpr_key: str, tpr_key: str):
        fp_rates = []
        tp_rates = []
        for k_fold in model_results:
            for member in k_fold:
                false_pos_rate = member[fpr_key]
                true_pos_rate = member[tpr_key]
                tp_rates.append(true_pos_rate)
                fp_rates.append(false_pos_rate)
        return fp_rates, tp_rates

    def __interp_points(
        self,
        false_pos_rates: list[float],
        true_pos_rates: list[float],
        thresholds: list[float],
    ):
        """
        To compute points at various thresholds
        """
        for val in thresholds:
            point = np.interp(val, false_pos_rates, true_pos_rates)
            print("{} : {}".format(val, point))

    def __interpolate_internal(
        self, false_pr: list[float], true_pr: list[float], points: int
    ):
        min_arr_x = min(false_pr)
        max_arr_x = max(false_pr)
        new_arr_x = np.linspace(min_arr_x, max_arr_x, points)
        new_arr_y = np.interp(new_arr_x, false_pr, true_pr)
        return new_arr_x, new_arr_y

    def __interpolate(
        self, false_pr: list[list[float]], true_pr: list[list[float]], n_points=1000
    ):
        """
        Process the interpolation of multirounded k-fold (5-round 3-fold)
        cross-validation of true positive rate (TPR) and false positive rate (FPR) records
        """
        collector = []
        for fp, tp in zip(false_pr, true_pr):
            retval = self.__interpolate_internal(fp, tp, n_points)
            collector.append(retval)
        midx = [np.mean([c[0][i] for c in collector]) for i in range(n_points)]
        midy = [np.mean([c[1][i] for c in collector]) for i in range(n_points)]
        plt.plot(midx, midy)
        ## To obtain the values at various thresholds
        # points = [0.001, 0.01, 0.1]
        # self.__interp_points(midx, midy, points)

    def __get_model_names(self):
        return ["Phishing Email Classifier", "Phishing Email Classifier with NER"]

    def __plot_per_month_roc(self, outfile: str):
        keys = self.__get_model_names()
        plt.legend(keys, loc="lower right")
        plt.title(
            "ROC Curves: Phishing Detection ({} 2022)".format(
                "March" if "March" in outfile else "April"
            )
        )
        self.__common_plot(outfile)

    def __common_plot(self, outfile: str):
        plt.grid()
        plt.xlabel("False Positive Rate (Logarithmic Scale)")
        plt.ylabel("True Positive Rate")
        plt.xscale("log")
        plt.xlim([0.001, 1])
        plt.ylim([0, 1])
        plt.savefig("{}/{}.pdf".format(self._args.outdir, outfile), bbox_inches="tight")
        plt.clf()

    def __plot_train_test_roc(self, outfile: str):
        keys = self.__get_model_names()
        plt.legend(keys, loc="lower right")
        plt.title(
            "ROC Curves: Phishing Detection (Train set - March 2022, Test set - April 2022)"
        )
        self.__common_plot(outfile)


class StatisticsExperimentVisualizer:
    """
    Class utilized for displaying probability distributions
    and Jensen-Shannon divergences plots
    - all together for March neg/pos, April neg/pos
    - pairwise
    """

    def __init__(self, args: argparse.Namespace):
        self._args = args
        self._plot_color_key = "color"
        self._type_key = "type"
        self._pairs = self.__get_pairs()
        self._filenames_map = self.__get_filenames_map()

    def visualize_probabilities(self):
        """
        Visualizes probability distributions,
        initially pairwise, then all together
        """
        all_records = {}
        for filename in self._pairs:
            values = self._pairs[filename]
            first = values["first"]
            second = values["second"]
            first_emails_data = self.__get_probability_distribution(first)
            second_emails_data = self.__get_probability_distribution(second)

            if first not in all_records:
                all_records[first] = first_emails_data
            if second not in all_records:
                all_records[second] = second_emails_data
            self.__plot_prob_distributions(
                filename, values, first_emails_data, second_emails_data
            )
        self.__plot_prob_distributions_alltogether(all_records)

    def visualize_divergence_alltogether(self):
        """
        Visualizes Jensen-Shannon divergences,
        initially pairwise, then all together
        """
        all_records = {}
        for name, pair in self._pairs.items():
            values = argparse.Namespace(
                outdir=self._args.outdir,
                occurrences=self._args.occurrences,
                positive=pair["first"],
                negative=pair["second"],
            )
            divergence = Divergence()
            divergence.load_data(name, values)
            divergence.run()
            with open("{}/{}.json".format(self._args.outdir, name), "r") as file:
                vals = json.load(file)
                self.__plot_jensen_divergence(vals, name)
                all_records[name] = vals
        self.__plot_jensen_divergence_all(all_records)

    def __plot_jensen_divergence_all(self, all_records: dict):
        range_ = np.arange(len(next(iter(all_records.values()))))
        self.__plot_style()
        fig = plt.figure(figsize=[14, 4])
        ax = fig.add_axes([0, 0, 1, 1])
        width = 0.2
        algn = "center"

        multiplier = -1.5
        for record in all_records:
            ax.bar(
                range_ + multiplier * width,
                all_records[record].values(),
                color=self._pairs[record][self._plot_color_key],
                width=width,
                align=algn,
            )
            multiplier += 1

        ax.set_xticks(range_)
        ax.set_xticklabels(all_records[record].keys())
        plt.xlabel("Named entities")
        plt.ylabel("Bits")
        title = "Jensen-Shannon Divergence for dataset pairs"

        formatted_names = [key.replace("_", " ") for key in self._pairs.keys()]
        plt.legend(formatted_names)
        plt.title(title)
        plt.savefig("{}/js-plot.pdf".format(self._args.outdir), bbox_inches="tight")

    def __plot_prob_distributions(
        self,
        filename: str,
        values: dict,
        positive_emails_data: dict,
        negative_emails_data: dict,
    ):

        range_ = np.arange(len(positive_emails_data))
        self.__plot_style()
        fig = plt.figure(figsize=[14, 4])
        ax = fig.add_axes([0, 0, 1, 1])
        width = 0.25

        ax.bar(
            range_ - width / 2,
            positive_emails_data.values(),
            width=width,
            color=self._filenames_map[values["first"]][self._plot_color_key],
            align="center",
        )
        ax.bar(
            range_ + width / 2,
            negative_emails_data.values(),
            width=width,
            color=self._filenames_map[values["second"]][self._plot_color_key],
            align="center",
        )

        ax.set_xticks(range_)
        ax.set_xticklabels(positive_emails_data.keys())

        plt.xlabel("Named entities")
        plt.ylabel("Probability")
        name = filename.replace("_", " ")
        values = name.split()
        title = "Named Entity Probability Distributions ({})".format(name)
        plt.legend([" ".join(values[0:2]), " ".join(values[3:])])
        plt.title(title)
        plt.savefig(
            "{}/prob_{}.pdf".format(self._args.outdir, filename), bbox_inches="tight"
        )

    def __plot_jensen_divergence(self, data, outfile: str):
        range_ = np.arange(len(data.values()))
        fig = plt.figure(figsize=[14, 4])
        self.__plot_style()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.bar(
            range_,
            data.values(),
            width=0.5,
            align="center",
            color=self._pairs[outfile][self._plot_color_key],
        )
        ax.set_xticks(range_)
        ax.set_xticklabels(data.keys())

        plt.xlabel("Named entities")
        plt.ylabel("Bits")
        title = outfile.replace("_", " ")
        plt.title("Jensen-Shannon Divergence ({})".format(title))
        filename = "{}/js-{}.pdf".format(self._args.outdir, outfile)
        plt.savefig(filename, bbox_inches="tight")

    def __plot_prob_distributions_alltogether(self, all_records: dict):
        range_ = np.arange(len(all_records[next(iter(all_records))].keys()))
        self.__plot_style()
        fig = plt.figure(figsize=[14, 4])
        ax = fig.add_axes([0, 0, 1, 1])
        width = 0.2
        algn = "center"
        # Pos mar, pos Apr, neg Mar, neg Apr

        multiplier = -1.5
        legend_storage = []
        for record in all_records:
            legend_storage.append(self._filenames_map[record][self._type_key])
            ax.bar(
                range_ + multiplier * width,
                all_records[record].values(),
                color=self._filenames_map[record][self._plot_color_key],
                width=width,
                align=algn,
            )
            multiplier += 1

        ax.set_xticks(range_)
        ax.set_xticklabels(all_records[record].keys())

        plt.xlabel("Named entities")
        plt.ylabel("Probability")
        title = "Probability distributions"
        plt.legend(legend_storage)
        plt.title(title)
        plt.savefig(
            "{}/prob-dist-plot.pdf".format(self._args.outdir), bbox_inches="tight"
        )

    def __get_probability_distribution(self, filename: str):
        data = {}
        sent_key = "sentences"
        with open(filename, "r") as file:
            contents = json.load(file)
            for model in contents:
                json_entities = contents[model]
                normalization_v = 0
                for entity_type in json_entities:
                    if entity_type == sent_key:
                        continue
                    occurrences_dict = json_entities[entity_type]
                    normalization_v += sum(occurrences_dict.values())
                for entity_type in json_entities:
                    if entity_type == sent_key:
                        continue
                    occurrences_dict = json_entities[entity_type]
                    data[entity_type] = sum(occurrences_dict.values()) / normalization_v
        return data

    def __plot_style(self):
        plt.style.use("seaborn-deep")
        plt.grid()

    def __get_filenames_map(self):
        return {
            "data/march_negatives_stats.json": {
                self._plot_color_key: "forestgreen",
                self._type_key: "negative March",
            },
            "data/april_negatives_stats.json": {
                self._plot_color_key: "lightgreen",
                self._type_key: "negative April",
            },
            "data/march_positives_stats.json": {
                self._plot_color_key: "darkorange",
                self._type_key: "positive March",
            },
            "data/april_positives_stats.json": {
                self._plot_color_key: "red",
                self._type_key: "positive April",
            },
        }

    def __get_pairs(self):
        return {  # for simplicity key value pairs
            "negative_March_x_negative_April": {
                self._plot_color_key: "forestgreen",
                "first": "data/march_negatives_stats.json",
                "second": "data/april_negatives_stats.json",
            },
            "negative_March_x_positive_March": {
                self._plot_color_key: "gold",
                "first": "data/march_negatives_stats.json",
                "second": "data/march_positives_stats.json",
            },
            "negative_April_x_positive_April": {
                self._plot_color_key: "orange",
                "first": "data/april_negatives_stats.json",
                "second": "data/april_positives_stats.json",
            },
            "positive_March_x_positive_April": {
                self._plot_color_key: "red",
                "first": "data/march_positives_stats.json",
                "second": "data/april_positives_stats.json",
            },
        }
