import itertools
import warnings
from collections.abc import Iterable, Sequence
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from msreport.qtable import Qtable

from .style import ColorWheelDict, with_active_style


@with_active_style
def replicate_ratios(
    qtable: Qtable,
    exclude_invalid: bool = True,
    xlim: Iterable[float] = (-2, 2),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Figure to compare the similarity of expression values between replicates.

    Displays the distribution of pair-wise log2 ratios between samples of the same
    experiment. Comparisons of the same experiment are placed in the same row. Requires
    log2 transformed expression values.

    Args:
        qtable: A `Qtable` instance, which data is used for plotting.
        exclude_invalid: If True, rows are filtered according to the Boolean entries of
            the "Valid" column.
        xlim: Specifies the displayed range for the log2 ratios on the x-axis. Default
            is from -2 to 2.

    Returns:
        A matplotlib Figure and a list of Axes objects, containing the comparison plots.
    """
    tag: str = "Expression"
    table = qtable.make_sample_table(
        tag, samples_as_columns=True, exclude_invalid=exclude_invalid
    )
    design = qtable.get_design()

    color_wheel = ColorWheelDict()
    for exp in design["Experiment"].unique():
        _ = color_wheel[exp]

    experiments = []
    for experiment in design["Experiment"].unique():
        if len(qtable.get_samples(experiment)) >= 2:
            experiments.append(experiment)
    if not experiments:
        fig, ax = plt.subplots(1, 1, figsize=(2, 1.3))
        fig.suptitle("Pair wise comparison of replicates", y=1.1)
        ax.text(0.5, 0.5, "No replicate\ndata available", ha="center", va="center")
        ax.grid(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        sns.despine(top=False, right=False, fig=fig)
        return fig, np.array([ax])

    num_experiments = len(experiments)
    max_replicates = max([len(qtable.get_samples(exp)) for exp in experiments])
    max_combinations = len(list(itertools.combinations(range(max_replicates), 2)))

    suptitle_space_inch = 0.55
    ax_height_inch = 0.6
    ax_width_inch = 1.55
    ax_hspace_inch = 0.35
    fig_height = (
        num_experiments * ax_height_inch
        + (num_experiments - 1) * ax_hspace_inch
        + suptitle_space_inch
    )
    fig_width = max_combinations * ax_width_inch
    fig_size = (fig_width, fig_height)

    subplot_top = 1 - (suptitle_space_inch / fig_height)
    subplot_hspace = ax_hspace_inch / ax_height_inch

    fig, axes = plt.subplots(
        num_experiments, max_combinations, figsize=fig_size, sharex=True
    )
    if num_experiments == 1 and max_combinations == 1:
        axes = np.array([[axes]])
    elif num_experiments == 1:
        axes = np.array([axes])
    elif max_combinations == 1:
        axes = np.array([axes]).T
    fig.subplots_adjust(
        bottom=0, top=subplot_top, left=0, right=1, hspace=subplot_hspace
    )
    fig.suptitle("Pair wise comparison of replicates", y=1)

    for x_pos, experiment in enumerate(experiments):
        sample_combinations = itertools.combinations(qtable.get_samples(experiment), 2)
        for y_pos, (s1, s2) in enumerate(sample_combinations):
            s1_label = design.loc[(design["Sample"] == s1), "Replicate"].tolist()[0]
            s2_label = design.loc[(design["Sample"] == s2), "Replicate"].tolist()[0]
            ax = axes[x_pos, y_pos]
            ratios = table[s1] - table[s2]
            ratios = ratios[np.isfinite(ratios)]
            ylabel = experiment if y_pos == 0 else ""
            title = f"{s1_label} vs {s2_label}"
            color = color_wheel[experiment]

            sns.kdeplot(x=ratios, fill=True, ax=ax, zorder=3, color=color, alpha=0.5)
            ax.set_title(title, fontsize=plt.rcParams["axes.labelsize"])
            ax.set_ylabel(ylabel, rotation=0, va="center", ha="right")
            ax.set_xlabel("Ratio [log2]")
            ax.tick_params(labelleft=False)
            ax.locator_params(axis="x", nbins=5)

    axes[0, 0].set_xlim(xlim)
    for ax in axes.flatten():
        if not ax.has_data():
            ax.remove()
            continue

        ax.axvline(x=0, color="#999999", lw=1, zorder=2)
        ax.grid(False, axis="y")
    sns.despine(top=True, right=True, fig=fig)

    return fig, axes


@with_active_style
def experiment_ratios(
    qtable: Qtable,
    experiments: Optional[str] = None,
    exclude_invalid: bool = True,
    ylim: Sequence[float] = (-2, 2),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Figure to compare the similarity of expression values between experiments.

    Intended to evaluate the bulk distribution of expression values after normalization.
    For each experiment a subplot is generated, which displays the distribution of log2
    ratios to a pseudo reference experiment as a density plot. The pseudo reference
    values are calculated as the average intensity values of all experiments. Only rows
    with quantitative values in all experiment are considered.

    Requires "Events experiment" columns and that average experiment expression values
    are calculated. This can be achieved by calling
    `msreport.analyze.analyze_missingness(qtable: Qtable)` and
    `msreport.analyze.calculate_experiment_means(qtable: Qtable)`.

    Args:
        qtable: A `Qtable` instance, which data is used for plotting.
        experiments: Optional, list of experiments that will be displayed. If None, all
            experiments from `qtable.design` will be used.
        exclude_invalid: If True, rows are filtered according to the Boolean entries of
            the "Valid" column.
        ylim: Specifies the displayed range for the log2 ratios on the y-axis. Default
            is from -2 to 2.

    Raises:
        ValueError: If only one experiment is specified in the `experiments` parameter
            or if the specified experiments are not present in the qtable design.

    Returns:
        A matplotlib Figure and a list of Axes objects, containing the comparison plots.
    """
    tag: str = "Expression"

    if experiments is not None and len(experiments) == 1:
        raise ValueError(
            "Only one experiment is specified, please provide at least two experiments."
        )
    elif experiments is not None:
        experiments_not_in_design = set(experiments) - set(qtable.design["Experiment"])
        if experiments_not_in_design:
            raise ValueError(
                "All experiments must be present in qtable.design. The following "
                f"experiments are not present: {experiments_not_in_design}"
            )
    else:
        experiments = qtable.design["Experiment"].unique().tolist()

    if len(experiments) < 2:
        fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.3))
        fig.suptitle("Comparison of experiments means", y=1.1)
        ax.text(
            0.5,
            0.5,
            "Comparison not possible.\nOnly one experiment\npresent in design.",
            ha="center",
            va="center",
        )
        ax.grid(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        sns.despine(top=False, right=False, fig=fig)
        return fig, np.array([ax])

    sample_data = qtable.make_sample_table(tag, samples_as_columns=True)
    experiment_means = {}
    for experiment in experiments:
        samples = qtable.get_samples(experiment)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            row_means = np.nanmean(sample_data[samples], axis=1)
        experiment_means[experiment] = row_means
    experiment_data = pd.DataFrame(experiment_means)

    # Only consider rows with quantitative values in all experiments
    mask = np.all([(qtable.data[f"Events {exp}"] > 0) for exp in experiments], axis=0)
    if exclude_invalid:
        mask = mask & qtable["Valid"]
    experiment_data = experiment_data[mask]
    pseudo_reference = np.nanmean(experiment_data, axis=1)
    ratio_data = experiment_data.subtract(pseudo_reference, axis=0)

    color_wheel = ColorWheelDict()
    for exp in qtable.design["Experiment"].unique():
        _ = color_wheel[exp]
    num_experiments = len(experiments)

    suptitle_space_inch = 0.55
    ax_height_inch = 1.25
    ax_width_inch = 0.65
    ax_wspace_inch = 0.2
    fig_height = ax_height_inch + suptitle_space_inch
    fig_width = num_experiments * ax_width_inch + (num_experiments - 1) * ax_wspace_inch
    fig_size = (fig_width, fig_height)

    subplot_top = 1 - (suptitle_space_inch / fig_height)
    subplot_wspace = ax_wspace_inch / ax_width_inch

    fig, axes = plt.subplots(1, num_experiments, figsize=fig_size, sharey=True)
    fig.subplots_adjust(
        bottom=0, top=subplot_top, left=0, right=1, wspace=subplot_wspace
    )
    fig.suptitle("Comparison of experiments means", y=1)

    for exp_pos, experiment in enumerate(experiments):
        ax = axes[exp_pos]
        values = ratio_data[experiment]
        color = color_wheel[experiment]
        sns.kdeplot(y=values, fill=True, ax=ax, zorder=3, color=color, alpha=0.5)
        if exp_pos == 0:
            ax.set_title(
                f"n={str(len(values))}",
                fontsize=plt.rcParams["xtick.labelsize"],
                loc="left",
            )
        ax.tick_params(labelbottom=False)
        ax.set_xlabel(experiment, rotation=90)

    axes[0].set_ylabel("Ratio [log2]\nto pseudo reference")
    axes[0].set_ylim(ylim)
    for ax in axes:
        ax.axhline(y=0, color="#999999", lw=1, zorder=2)
        ax.grid(False, axis="x")
    sns.despine(top=True, right=True, fig=fig)
    return fig, axes
