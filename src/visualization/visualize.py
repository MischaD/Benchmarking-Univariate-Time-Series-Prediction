import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import pandas as pd
import time
import seaborn as sns

from log import logger_viz
from .utils import FAU_BLUE, FAU_GREY, TURQUOIS


class Visualization:
    """Visualization base class for consistent plots.
    """
    def __init__(self, use_latex_font=False, font_scale=1):
        """

        :param use_latex_font: make matplotlib use latex font
        :param font_scale: scale font, same as seaborn font scale
        """

        if use_latex_font:
            logger_viz.info("using latex font")
            plt.rcParams.update({
                "text.usetex": True,
                "font.size": 20,
                "font.family": "serif",
                "font.sans-serif": ["Palatino"]
            })

        mpl.style.use('seaborn')
        sns.set(font="serif", font_scale=font_scale)

        self.fig = None
        self.axs = None
        self.plt = plt

    def subplots(self, nrows, ncols, squeeze=False, *args, **kwargs):
        self.fig, self.axs = plt.subplots(nrows,
                                          ncols,
                                          squeeze=squeeze,
                                          *args,
                                          **kwargs)
        return self.fig, self.axs

    def get_subplot_count(self):
        return len(self.axs) * len(self.axs[0])

    def save(self, filename):
        self.plt.savefig(filename + ".pdf")

    def get_ax(self, i: int):
        """get axis that corresponds to the integer value. Silently ignores if the requested value is too high"""
        nrows, ncols = len(self.axs), len(self.axs[0])
        if i >= nrows * ncols:
            logger_viz.warn(f"get ax parameter i out of range for i = {i}")
            return self.axs[-1, -1]

        row = i // ncols
        col = i % ncols
        return self.axs[row, col]


class InputOutputVisualization(Visualization):
    """
    Visualize input and output relationship.
    """
    def plot(self, time_series: np.ndarray, pred_starts: list):
        if len(self.axs) * len(self.axs[0]) != len(time_series):
            logger_viz.warning(
                f"Unexpected amount of time series for {self.__class__}")

        for i, ts in enumerate(time_series):
            x = np.arange(len(time_series[i]))
            ax = self.get_ax(i)
            ax.plot(x[:pred_starts[i] + 1],
                    ts[:pred_starts[i] + 1],
                    '--o',
                    color=FAU_BLUE,
                    label="Input")
            ax.plot(x[pred_starts[i]:],
                    ts[pred_starts[i]:],
                    '--',
                    marker="x",
                    markeredgewidth=2,
                    color=TURQUOIS,
                    label="Output")

    def add_vlines(self, vlines, min_val, max_val):
        for i, vlines_ in enumerate(vlines):
            ax = self.get_ax(i)
            ax.vlines(vlines_, min_val, max_val, 'black', linestyles="dashed")

    def add_subplot_titles(self, titles):
        for i, title in enumerate(titles):
            self.get_ax(i).set_title(title)

    def add_subplot_xylabel(self, x_labels, y_labels):
        if isinstance(x_labels, str):
            x_labels = self.get_subplot_count() * [
                x_labels,
            ]
        if isinstance(y_labels, str):
            y_labels = self.get_subplot_count() * [
                y_labels,
            ]

        for i, x_label in enumerate(x_labels):
            self.get_ax(i).set_xlabel(x_label)
        for i, y_label in enumerate(y_labels):
            self.get_ax(i).set_ylabel(y_label)

    def adjust_bottom_margin(self, margin):
        plt.subplots_adjust(hspace=margin)

    def add_legend(self):
        for i in range(self.get_subplot_count()):
            ax = self.get_ax(i)
            ax.legend()

    def quick_plot(self, sample, pred_start, title, save_to, figsize=(16, 9)):
        """Creates a very basic plot of a single sample

        :param sample: One dimensional array representing a time series
        :param pred_start: Cut off point where the prediction starts
        :param title: Title of the figure
        :param save_to: Title of the file to save the image to (without filename extension)
        :param figsize: Figsize, default (16, 9)
        :return:
        """
        assert sample.ndim == 1
        self.subplots(nrows=1, ncols=1, figsize=figsize)
        self.plot([
            sample,
        ], pred_starts=[
            pred_start,
        ])
        self.fig.suptitle(title)
        self.plt.grid()
        self.save(filename=save_to)


class ModelPredictionVisualization(Visualization):
    """
    Visualization of model prediction.
    """
    def plot(self, model_inputs, time_series_real, time_series_predicted,
             pred_starts):
        time_series_predicted_padded = np.zeros_like(time_series_real)
        for i, ts_pred in enumerate(time_series_predicted):
            #if len(ts_r) == len(ts_pred):
            #    time_series_predicted_padded[i] = time_series_real
            #else:
            time_series_predicted_padded[i, :len(ts_pred)] = ts_pred

        ts_real = np.concatenate((model_inputs, time_series_real), axis=1)
        ts_pred = np.concatenate((model_inputs, time_series_predicted_padded),
                                 axis=1)

        for i, ts in enumerate(model_inputs):
            x = np.arange(len(ts_real[i]))
            ax = self.get_ax(i)
            ax.plot(x[:pred_starts[i] + 1],
                    ts_real[i][:pred_starts[i] + 1],
                    '-o',
                    color=FAU_BLUE,
                    label="Input")
            ax.plot(x[pred_starts[i]:],
                    ts_real[i][pred_starts[i]:],
                    '--o',
                    color=TURQUOIS,
                    label="Output actual")
            ax.plot(x[pred_starts[i]:],
                    ts_pred[i][pred_starts[i]:],
                    '--',
                    marker="x",
                    markeredgewidth=2,
                    color=FAU_GREY,
                    label="Output predicted")


class ValidationLossVisualization(Visualization):
    """
    Validation loss curve visualization
    """
    def plot(self, steps, val_losses, labels, xlabel, ylabel):
        colors = [FAU_BLUE, FAU_GREY, TURQUOIS]
        markers = [".", "*", "X"]
        assert len(val_losses) <= 3
        assert len(labels) <= 3

        for i in range(len(val_losses)):
            ax = self.get_ax(0)
            ax.plot(
                steps[i],
                val_losses[i],
                label=labels[i],
                color=colors[i],
                marker=markers[i],
            )
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
        self.plt.grid()


class TrainValidationLossVisualization(Visualization):
    """
    Train and validation loss curve visualization.
    """
    def plot(self,
             steps,
             val_losses,
             labels,
             linestyles,
             xlabel,
             ylabel,
             colors=None,
             markers=None):
        if colors is None:
            colors = [
                FAU_BLUE, FAU_BLUE, FAU_GREY, FAU_GREY, TURQUOIS, TURQUOIS
            ]
        if markers is None:
            markers = [".", ".", "*", "*", "X", "X"]

        for i in range(len(val_losses)):
            ax = self.get_ax(0)
            ax.plot(
                steps[i],
                val_losses[i],
                label=labels[i],
                linestyle=linestyles[i],
                color=colors[i],
                marker=markers[i],
            )
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
        self.plt.grid()


class ConfidenceIntervalVisualization(Visualization):
    """Visualization of 95% confidence intervals.

    Based on the idea of A Comprehensive Analysis of Deep Regression by Lathuiliere et al.
    """
    MODEL_NAMES = ["Transformer", "CNN",
                   "LSTM"]  # will be plotted --> capital letters

    def plot(self,
             predictions,
             characteristic,
             xlabel,
             ylabel,
             colors,
             verbose=False,
             plot_only_median=False):
        median_confidence_intervals = []
        char = sorted(predictions[characteristic].unique())
        for model in self.MODEL_NAMES:
            if verbose:
                print(f"{model}:")

            model_confidence = []
            for c in char:
                rel_preds = predictions[(predictions[characteristic] == c)
                                        & (predictions["model"] == model)]
                n = len(rel_preds)
                lower_idx = int(
                    np.ceil(n * 0.5 - 1.96 * np.sqrt(n * 0.5 * 0.5)))
                median_idx = n // 2
                upper_idx = int(
                    np.ceil(n * 0.5 + 1.96 * np.sqrt(n * 0.5 * 0.5)))

                sorted_trafo_preds = rel_preds.sort_values(by="mae")
                lower = sorted_trafo_preds.iloc[lower_idx]
                median = sorted_trafo_preds.iloc[median_idx]
                upper = sorted_trafo_preds.iloc[upper_idx]

                model_confidence.append(
                    [lower['mae'], median['mae'], upper['mae']])

                if verbose:
                    print(
                        f"f: {c:3n} lower: {lower['mae']:1.3f} - median: {median['mae']:1.3f} - upper: {upper['mae']:1.3f}"
                    )

            median_confidence_intervals.append(model_confidence)

        meds = np.array(median_confidence_intervals)
        ax = self.get_ax(0)
        for i, model in enumerate(self.MODEL_NAMES):
            ax.plot(char, meds[i, :, 1], label=model, color=colors[i])
            if not plot_only_median:
                ax.fill_between(char,
                                meds[i, :, 0],
                                meds[i, :, 2],
                                color=colors[i],
                                alpha=0.3)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)


class InferenceTimeVisualization(Visualization):
    """Visualizes inference time in seconds"""
    def _do_test_runs(self, n_runs, trainer, model):
        start = time.time()
        for _ in range(n_runs):
            trainer.test(model=model,
                         test_dataloaders=self.dataloader,
                         verbose=False)
        stop = time.time()
        return stop - start

    def do_inference(self, n_runs, cnn_model, cnn_trainer, lstm_model,
                     lstm_trainer, trafo_model, trafo_trainer, dataloader):
        self.batch_size = dataloader.batch_size
        self.n_runs = n_runs
        self.dataloader = dataloader

        self.cnn_time = self._do_test_runs(n_runs, cnn_trainer, cnn_model)
        self.lstm_time = self._do_test_runs(n_runs, lstm_trainer, lstm_model)
        self.trafo_time = self._do_test_runs(n_runs, trafo_trainer,
                                             trafo_model)

    def plot(self, ax, palette):
        df = pd.DataFrame({
            "model": ["Transformer", "CNN", "LSTM"],
            "exectime": [self.trafo_time, self.cnn_time, self.lstm_time]
        })
        df["exectime"] = df["exectime"] / (self.n_runs * len(self.dataloader))
        df = df.sort_values("exectime", ascending=False)
        sns.barplot(x="exectime", y="model", data=df, color=palette[0], ax=ax)
