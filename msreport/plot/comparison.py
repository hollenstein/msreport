import itertools
import warnings
from collections.abc import Iterable, Sequence
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import msreport.helper
from msreport.qtable import Qtable

from ._partial_plots import annotated_scatter
from .style import with_active_style


@with_active_style
def volcano_ma(
    qtable: Qtable,
    experiment_pair: Iterable[str],
    comparison_tag: str = " vs ",
    pvalue_tag: str = "P-value",
    special_entries: Optional[list[str]] = None,
    special_proteins: Optional[list[str]] = None,
    annotation_column: str = "Gene name",
    exclude_invalid: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Generates a volcano and an MA plot for the comparison of two experiments.

    Args:
        qtable: A `Qtable` instance, which data is used for plotting.
        experiment_pair: The names of the two experiments that will be compared,
            experiments must be present in qtable.design.
        comparison_tag: String used in comparison columns to separate a pair of
            experiments; default " vs ", which corresponds to the MsReport convention.
        pvalue_tag: String used for matching the pvalue columns; default "P-value",
            which corresponds to the MsReport convention.
        special_entries: Optional, allows to specify a list of entries from the
            `qtable.id_column` column to be annotated.
        special_proteins: This argument is deprecated, use 'special_entries' instead.
        annotation_column: Column used for labeling the points of special entries in the
            scatter plot. Default "Gene name". If the 'annotation_column' is not present
            in the `qtable.data` table, the `qtable.id_column` is used instead.
        exclude_invalid: If True, rows are filtered according to the Boolean entries of
            the "Valid" column.

    Raises:
        ValueError: If the 'pvalue_tag', "Average expression" or "Ratio [log2]" column
            is missing in the Qtable for the specified experiment_pair.

    Returns:
        A matplotlib Figure object and a list of two Axes objects containing the volcano
        and the MA plot.
    """
    ratio_tag = "Ratio [log2]"
    expression_tag = "Average expression"
    comparison_group = comparison_tag.join(experiment_pair)

    for tag in [ratio_tag, expression_tag, pvalue_tag]:
        tag_column = msreport.helper.find_sample_columns(
            qtable.data, comparison_group, [tag]
        )
        if not tag_column:
            raise ValueError(
                f"Missing the required '{tag}' column for the comparison group "
                f"'{comparison_group}' in the Qtable."
            )

    if special_entries is None:
        special_entries = []
    if special_proteins is not None:
        warnings.warn(
            "The argument 'special_proteins' is deprecated, use 'special_entries' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        special_entries = list(special_entries) + list(special_proteins)

    data = qtable.get_data(exclude_invalid=exclude_invalid)
    if annotation_column not in data.columns:
        annotation_column = qtable.id_column

    scatter_size = 2 / (max(min(data.shape[0], 10000), 1000) / 1000)

    masks = {
        "highlight": data[qtable.id_column].isin(special_entries),
        "default": ~data[qtable.id_column].isin(special_entries),
    }
    params = {
        "highlight": {
            "s": 10,
            "color": "#E73C40",
            "edgecolor": "#000000",
            "lw": 0.2,
            "zorder": 3,
        },
        "default": {"s": scatter_size, "color": "#40B7B5", "zorder": 2},
    }

    for column in msreport.helper.find_sample_columns(
        data, pvalue_tag, [comparison_group]
    ):
        data[column] = np.log10(data[column]) * -1

    suptitle_space_inch = 0.4
    ax_height_inch = 3.2
    ax_width_inch = 3.2
    ax_wspace_inch = 1

    fig_height = suptitle_space_inch + ax_height_inch
    fig_width = ax_width_inch * 2 + ax_wspace_inch
    fig_size = (fig_width, fig_height)

    subplot_top = 1 - (suptitle_space_inch / fig_height)
    subplot_wspace = ax_wspace_inch / ax_width_inch

    fig, axes = plt.subplots(1, 2, figsize=fig_size, sharex=True)
    fig.subplots_adjust(
        bottom=0, top=subplot_top, left=0, right=1, wspace=subplot_wspace
    )
    fig.suptitle(f"Volcano and MA plot of: {comparison_group}", y=1)

    for ax, x_variable, y_variable in [
        (axes[0], ratio_tag, pvalue_tag),
        (axes[1], ratio_tag, expression_tag),
    ]:
        x_col = " ".join([x_variable, comparison_group])
        y_col = " ".join([y_variable, comparison_group])
        x_values = data[x_col]
        y_values = data[y_col]
        xy_labels = data[annotation_column]

        valid_values = np.isfinite(x_values) & np.isfinite(y_values)
        mask_default = masks["default"] & valid_values
        mask_special = masks["highlight"] & valid_values

        ax.scatter(x_values[mask_default], y_values[mask_default], **params["default"])
        annotated_scatter(
            x_values=x_values[mask_special],
            y_values=y_values[mask_special],
            labels=xy_labels[mask_special],
            ax=ax,
            scatter_kws=params["highlight"],
        )

        ax.set_xlabel(x_variable)
        if y_variable == pvalue_tag:
            ax.set_ylabel(f"{y_variable} [-log10]")
        else:
            ax.set_ylabel(f"{y_variable} [log2]")
        ax.grid(axis="both", linestyle="dotted")

    return fig, axes


@with_active_style
def expression_comparison(
    qtable: Qtable,
    experiment_pair: list[str],
    comparison_tag: str = " vs ",
    plot_average_expression: bool = False,
    special_entries: Optional[list[str]] = None,
    special_proteins: Optional[list[str]] = None,
    annotation_column: str = "Gene name",
    exclude_invalid: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Generates an expression comparison plot for two experiments.

    The subplot in the middle displays the average expression of the two experiments on
    the y-axis and the log fold change on the x-axis. The subplots on the left and right
    display entries with only missing values in one of the two experiments.

    Args:
        qtable: A `Qtable` instance, which data is used for plotting.
        experiment_pair: The names of the two experiments that will be compared,
            experiments must be present in qtable.design.
        comparison_tag: String used in comparison columns to separate a pair of
            experiments; default " vs ", which corresponds to the MsReport convention.
        plot_average_expression: If True plot average expression instead of maxium
            expression. Default False.
        special_entries: Optional, allows to specify a list of entries from the
            `qtable.id_column` column to be annotated.
        special_proteins: This argument is deprecated, use 'special_entries' instead.
        annotation_column: Column used for labeling the points of special entries in the
            scatter plot. Default "Gene name". If the 'annotation_column' is not present
            in the `qtable.data` table, the `qtable.id_column` is used instead.
        exclude_invalid: If True, rows are filtered according to the Boolean entries of
            the "Valid" column.

    Raises:
        ValueError: If the "Expression" and "Events" columns for the specified
            experiments are missing in the Qtable.

    Returns:
        A matplotlib Figure objects and a list of three Axes objects containing the
        expression comparison plots.
    """
    missing_columns = []
    for exp in experiment_pair:
        for tag in ["Expression", "Events"]:
            if f"{tag} {exp}" not in qtable.data.columns:
                missing_columns.append(f"{tag} {exp}")
    missing_columns = sorted(missing_columns)
    if missing_columns:
        raise ValueError(
            f"Missing the required columns in the Qtable: {missing_columns}."
        )

    if special_entries is None:
        special_entries = []
    if special_proteins is not None:
        warnings.warn(
            "The argument 'special_proteins' is deprecated, use 'special_entries' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        special_entries = list(special_entries) + list(special_proteins)

    exp_1, exp_2 = experiment_pair
    comparison_group = comparison_tag.join(experiment_pair)

    qtable_data = qtable.get_data(exclude_invalid=exclude_invalid)
    if annotation_column not in qtable_data.columns:
        annotation_column = qtable.id_column
    total_scatter_area = 5000
    params = {
        "highlight": {
            "s": 10,
            "color": "#E73C40",
            "edgecolor": "#000000",
            "lw": 0.2,
            "zorder": 3,
        },
        "default": {"alpha": 0.75, "color": "#40B7B5", "zorder": 2},
    }

    mask = (qtable_data[f"Events {exp_1}"] + qtable_data[f"Events {exp_2}"]) > 0
    qtable_data = qtable_data[mask]

    only_exp_1 = qtable_data[f"Events {exp_2}"] == 0
    only_exp_2 = qtable_data[f"Events {exp_1}"] == 0
    mask_both = np.invert(np.any([only_exp_1, only_exp_2], axis=0))

    # Test if plotting maximum intensity is better than average
    qtable_data[f"Maximum expression {comparison_group}"] = np.max(
        [qtable_data[f"Expression {exp_2}"], qtable_data[f"Expression {exp_1}"]], axis=0
    )
    qtable_data[f"Average expression {comparison_group}"] = np.nanmean(
        [qtable_data[f"Expression {exp_2}"], qtable_data[f"Expression {exp_1}"]], axis=0
    )

    def scattersize(df: pd.DataFrame, total_area) -> float:
        if len(values) > 0:
            size = min(max(np.sqrt(total_area / df.shape[0]), 0.5), 4)
        else:
            size = 1
        return size

    suptitle_space_inch = 0.4
    ax_height_inch = 3.2
    main_ax_width_inch = 3.2
    side_ax_width_inch = 0.65
    ax_wspace_inch = 0.6
    width_ratios = [side_ax_width_inch, main_ax_width_inch, side_ax_width_inch]

    fig_height = suptitle_space_inch + ax_height_inch
    fig_width = main_ax_width_inch + (side_ax_width_inch + ax_wspace_inch) * 2
    fig_size = (fig_width, fig_height)

    subplot_top = 1 - (suptitle_space_inch / fig_height)
    subplot_wspace = ax_wspace_inch / np.mean(width_ratios)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=fig_size,
        sharey=True,
        gridspec_kw={
            "bottom": 0,
            "top": subplot_top,
            "left": 0,
            "right": 1,
            "wspace": subplot_wspace,
            "width_ratios": width_ratios,
        },
    )
    fig.suptitle(f'Comparison of "Expression": {comparison_group}', y=1)

    # Plot values quantified in both experiments
    ax = axes[1]
    values = qtable_data[mask_both]
    s = scattersize(values, total_scatter_area)
    x_variable = "Ratio [log2]"
    y_variable = (
        "Average expression" if plot_average_expression else "Maximum expression"
    )
    x_col = " ".join([x_variable, comparison_group])
    y_col = " ".join([y_variable, comparison_group])
    x_values = values[x_col]
    y_values = values[y_col]
    ax.grid(axis="both", linestyle="dotted")
    ax.scatter(x_values, y_values, s=s, **params["default"])
    highlight_mask = values[qtable.id_column].isin(special_entries)
    annotated_scatter(
        x_values=x_values[highlight_mask],
        y_values=y_values[highlight_mask],
        labels=values[annotation_column][highlight_mask],
        ax=ax,
        scatter_kws=params["highlight"],
    )

    ax.set_xlabel(x_variable)
    ax.set_ylabel(y_variable)

    # Plot values quantified only in one experiment
    for ax, mask, exp in [(axes[2], only_exp_1, exp_1), (axes[0], only_exp_2, exp_2)]:
        y_variable = f"Expression {exp}"
        values = qtable_data[mask]
        highlight_mask = values[qtable.id_column].isin(special_entries)
        s = scattersize(values, total_scatter_area)

        ax.grid(axis="y", linestyle="dotted")
        ax.set_ylabel(y_variable)

        if len(values) == 0:
            continue

        sns.stripplot(
            y=values[y_variable],
            jitter=True,
            size=np.sqrt(s * 2),
            marker="o",
            edgecolor="none",
            ax=ax,
            **params["default"],
        )

        xlim = -0.2, 0.2
        ax.set_xlim(xlim)
        offsets = ax.collections[0].get_offsets()[highlight_mask]
        annotated_scatter(
            x_values=offsets[:, 0],
            y_values=offsets[:, 1],
            labels=values[annotation_column][highlight_mask],
            ax=ax,
            scatter_kws=params["highlight"],
        )
        ax.set_xlim(xlim)

    # Important to reverse the order here which experiment is on the left and right
    axes[0].set_xlabel(f"Absent in\n{exp_1}")
    axes[2].set_xlabel(f"Absent in\n{exp_2}")

    return fig, axes


@with_active_style
def pvalue_histogram(
    qtable: Qtable,
    pvalue_tag: str = "P-value",
    comparison_tag: str = " vs ",
    experiment_pairs: Optional[Sequence[Iterable[str]]] = None,
    exclude_invalid: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Generates p-value histograms for one or multiple experiment comparisons.

    Histograms are generated with 20 bins of size 0.05. The p-value distribution of each
    experiment comparison is shown with a separate subplot.

    Args:
        qtable: A `Qtable` instance, which data is used for plotting.
        pvalue_tag: String used for matching the pvalue columns; default "P-value",
            which corresponds to the MsReport convention.
        comparison_tag: String used in comparison columns to separate a pair of
            experiments; default " vs ", which corresponds to the MsReport convention.
        experiment_pairs: Optional, list of experiment pairs that will be used for
            plotting. For each experiment pair a p-value column must exists that follows
            the format f"{pvalue_tag} {experiment_1}{comparison_tag}{experiment_2}".
            If None, all experiment comparisons that are found in qtable.data are used.
        exclude_invalid: If True, rows are filtered according to the Boolean entries of
            the "Valid" column.

    Raises:
        ValueError: If no experiment pairs are found in the Qtable for the provided
            p-value tag and comparison tag or if any of the provided experiment pairs
            does not exist in the Qtable.

    Returns:
        A matplotlib Figure and a list of Axes objects, containing the p-value plots.
    """
    data = qtable.get_data(exclude_invalid=exclude_invalid)

    def _get_valid_experiment_pairs(
        pairs: Iterable[Iterable[str]],
    ) -> list[Iterable[str]]:
        valid_pairs = []
        for pair in pairs:
            comparison_group = comparison_tag.join(pair)
            comparison_column = f"{pvalue_tag} {comparison_group}"
            if comparison_column in data.columns:
                valid_pairs.append(pair)
        return valid_pairs

    # Find all experiment pairs
    if experiment_pairs is not None:
        valid_pairs = _get_valid_experiment_pairs(experiment_pairs)
        invalid_pairs = list(set(experiment_pairs) - set(valid_pairs))
        if invalid_pairs:
            raise ValueError(
                "The following provided experiment pairs do not exist in the Qtable: "
                f"{invalid_pairs}"
            )
    else:
        experiment_pairs = _get_valid_experiment_pairs(
            itertools.permutations(qtable.get_experiments(), 2)
        )
        if not experiment_pairs:
            raise ValueError(
                "No experiment pairs found in the Qtable for p-value tag "
                f"'{pvalue_tag}' and comparison tag '{comparison_tag}'."
            )

    num_plots = len(experiment_pairs)

    suptitle_space_inch = 0.4
    ax_height_inch = 1.8
    ax_width_inch = 1
    ax_wspace_inch = 0.6

    fig_width = num_plots * ax_width_inch + (num_plots - 1) * ax_wspace_inch
    fig_height = ax_height_inch + suptitle_space_inch
    fig_size = (fig_width, fig_height)

    subplot_top = 1 - (suptitle_space_inch / fig_height)
    subplot_wspace = ax_wspace_inch / ax_width_inch

    fig, axes = plt.subplots(1, num_plots, figsize=fig_size, sharex=True, sharey=True)
    if num_plots == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes)
    fig.subplots_adjust(
        bottom=0, top=subplot_top, left=0, right=1, wspace=subplot_wspace
    )
    fig.suptitle("P-value histogram of pair-wise experiment comparisons", y=1)

    bins = np.arange(0, 1.01, 0.05)
    for ax_pos, experiment_pair in enumerate(experiment_pairs):  # type: ignore
        comparison_group = comparison_tag.join(experiment_pair)
        comparison_column = f"{pvalue_tag} {comparison_group}"
        comparison_label = f"{comparison_tag}\n".join(experiment_pair)
        p_values = data[comparison_column]

        ax = axes[ax_pos]
        ax2 = ax.twinx()
        ax2.set_yticks([])
        ax2.set_ylabel(comparison_label)

        ax.hist(
            p_values,
            bins=bins,
            color=None,
            edgecolor="#215e5d",
            linewidth=1.5,
            zorder=2,
        )
        ax.hist(
            p_values,
            bins=bins,
            color="#40B7B5",
            edgecolor=None,
            linewidth=0,
            zorder=2.1,
        )

        ax.set_xticks([0, 0.5, 1])
        # Need to remove the ticks manually because creating the twin axis somehow
        # overrides the rcParams settings.
        ax.tick_params(
            left=plt.rcParams["ytick.left"], right=plt.rcParams["ytick.right"]
        )
        ax.set_xlabel(pvalue_tag)
        ax.grid(False, axis="x")
        sns.despine(top=True, right=True)

    axes[0].set_ylabel(f"{pvalue_tag} count")
    ax.set_xlim(-0.05, 1.05)

    return fig, axes
