from collections.abc import Iterable
from collections import UserDict
import itertools
from typing import Optional
import re
import warnings

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.preprocessing
import sklearn.decomposition

from msreport.qtable import Qtable
import msreport.qanalysis


def set_dpi(dpi: int) -> None:
    """Changes the default dots per inch settings.

    This effectively makes figures smaller or larger, without affecting the
    relative sizes of elements within the figures.
    """
    plt.rcParams["figure.dpi"] = dpi


class ColorWheelDict(UserDict):
    """Lookup dictionary that maps keys to hexcolors.

    When a key is not present the first color of the color wheel is added as
    the value, and the color is moved from the beginning to the end of the
    color wheel. If no list of colors is specified, a default list of ten
    colors is added to the color wheel. It is also possible to manually set
    key and color pairs by using the same syntax as for a regular dictionary.
    """

    def __init__(self, colors: Optional[list[str]] = None):
        self.data = {}

        if colors is not None:
            self.colors = colors
        else:
            self.colors = [
                "#80b1d3",
                "#fdb462",
                "#8dd3c7",
                "#bebada",
                "#fb8072",
                "#b3de69",
                "#fccde5",
                "#d9d9d9",
                "#bc80bd",
                "#ccebc5",
            ]
        self._color_wheel = self.colors.copy()

    def _next_color(self) -> str:
        color = self._color_wheel.pop(0)
        self._color_wheel.append(color)
        return color

    def __setitem__(self, key, value):
        is_hexcolor = re.search(r"^#(?:[0-9a-fA-F]{3}){1,2}$", value)
        if is_hexcolor:
            self.data[key] = value
        else:
            raise ValueError(f"the specified value {value} is not a hexcolor.")

    def __getitem__(self, key):
        if key not in self.data:
            self.data[key] = self._next_color()
        return self.data[key]


def missing_values_vertical(
    qtable: Qtable,
    remove_invalid: bool = True,
) -> (plt.Figure, plt.Axes):
    """Vertical bar plot to analyze the completeness of quantification.

    The expression columns are used to analyze the number of samples with
    missing values per experiment. This figure must be generated before
    imputing missing values.

    Args:
        qtable: msreport.qtable.Qtable instance, which data is used for
            plotting.
        remove_invalid: If true and the column 'Valid' is present, rows are
            filtered according to the boolean entries of 'Valid'.
    Returns:
        A matplotlib Figure and Axes object containing the missing values plot.
    """
    experiments = qtable.get_experiments()
    num_experiments = len(experiments)

    table = qtable.data.copy()
    if remove_invalid and "Valid" in qtable.data:
        table = table[qtable.data["Valid"]]

    barwidth = 0.8
    barcolors = ["#31A590", "#FAB74E", "#EB3952"]
    figwidth = (num_experiments * 1.2) + 0.5
    figsize = (figwidth, 3.5)
    xtick_labels = ["No missing", "Some missing", "All missing"]

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, num_experiments, figsize=figsize, sharey=True)
    for exp_num, exp in enumerate(experiments):
        ax = axes[exp_num]

        exp_missing = table[f"Missing {exp}"]
        exp_values = table[f"Events {exp}"]
        missing_none = (exp_missing == 0).sum()
        missing_some = ((exp_missing > 0) & (exp_values > 0)).sum()
        missing_all = (exp_values == 0).sum()

        y = [missing_none, missing_some, missing_all]
        x = range(len(y))
        ax.bar(x, y, width=barwidth, color=barcolors)
        if exp_num == 0:
            ax.set_ylabel("# Proteins")
        ax.set_title(exp)
        ax.set_xticks(np.array([0, 1, 2]) + 0.4)
        ax.set_xticklabels(xtick_labels, rotation=45, va="top", ha="right")
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#000000")
            ax.spines[spine].set_linewidth(1)
        ax.grid(False, axis="x")
        ax.grid(axis="y", linestyle="dashed", linewidth=1)
    sns.despine(top=True, right=True)
    fig.tight_layout()
    return fig, axes


def missing_values_horizontal(
    qtable: Qtable,
    remove_invalid: bool = True,
) -> (plt.Figure, plt.Axes):
    """Horizontal bar plot to analyze the completeness of quantification.

    The expression columns are used to analyze the number of samples with
    missing values per experiment. This figure must be generated before
    imputing missing values.

    Args:
        qtable: msreport.qtable.Qtable instance, which data is used for
            plotting.
        remove_invalid: If true and the column 'Valid' is present, rows are
            filtered according to the boolean entries of 'Valid'.

    Returns:
        A matplotlib Figure and Axes object containing the missing values plot.
    """
    experiments = qtable.get_experiments()
    num_experiments = len(experiments)

    table = qtable.data.copy()
    if remove_invalid and "Valid" in qtable.data:
        table = table[qtable.data["Valid"]]

    data = {"exp": [], "max": [], "some": [], "min": []}
    for exp in experiments:
        exp_missing = table[f"Missing {exp}"]
        total = len(exp_missing)
        num_replicates = len(qtable.get_samples(exp))
        missing_all = (exp_missing == num_replicates).sum()
        missing_none = (exp_missing == 0).sum()
        with_missing_some = total - missing_all

        data["exp"].append(exp)
        data["max"].append(total)
        data["some"].append(with_missing_some)
        data["min"].append(missing_none)

    plotheight = (num_experiments * 0.5) + 0.5
    legendheight = 1.5
    figheight = plotheight + legendheight
    figsize = (5, figheight)

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(y="exp", x="max", data=data, label="All missing", color="#EB3952")
    sns.barplot(y="exp", x="some", data=data, label="Some missing", color="#FAB74E")
    sns.barplot(y="exp", x="min", data=data, label="None missing", color="#31A590")

    ax.set_xlim(0, total)
    ax.set_title("Completeness of protein quantification per experiment")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1, 0), ncol=3)
    figure_space_for_legend = 1 - (legendheight / figheight)
    fig.tight_layout(rect=[0, 0, 1, figure_space_for_legend])
    return fig, ax


def sample_intensities(
    qtable: Qtable, tag: str = "Intensity", remove_invalid: bool = True
) -> (plt.Figure, plt.Axes):
    """Figure to compare the overall quantitative similarity of samples.

    Generates two subplots to compare the intensities of multiple samples. For
    the top subplot a pseudo reference sample is generated by calculating the
    average intensity values of all samples. For each row and sample the log2
    ratios to the pseudo reference are calculated. Only rows without missing
    values are selected and for each sample the log2 ratios to the pseudo
    reference are displayed as a box plots. The lower subplot displays the
    summed up intensities of all rows per sample as bar plots.

    Args:
        qtable: msreport.qtable.Qtable instance, which data is used for
            plotting.
        tag: String used for matching the intensity columns.
        remove_invalid: If true and the column 'Valid' is present, rows are
            filtered according to the boolean entries of 'Valid'.

    Returns:
        A matplotlib Figure and Axes object containing the intensity plots.
    """
    matrix = qtable.make_sample_matrix(tag, samples_as_columns=True)
    if remove_invalid and "Valid" in qtable.data:
        matrix = matrix[qtable.data["Valid"]]

    matrix = matrix.replace({0: np.nan})
    if msreport.helper.intensities_in_logspace(matrix):
        log2_matrix = matrix
        matrix = np.power(2, log2_matrix)
    else:
        log2_matrix = np.log2(matrix)
    samples = matrix.columns.tolist()

    finite_values = log2_matrix.isna().sum(axis=1) == 0
    pseudo_ref = np.nanmean(log2_matrix[finite_values], axis=1)
    log2_ratios = log2_matrix[finite_values].subtract(pseudo_ref, axis=0)

    bar_values = matrix.sum()
    box_values = [log2_ratios[c] for c in log2_ratios.columns]
    color_wheel = ColorWheelDict()
    colors = [color_wheel[exp] for exp in qtable.get_experiments(samples)]
    fig, axes = box_and_bars(box_values, bar_values, samples, colors=colors)
    axes[0].set_title(f'Comparison of "{tag}" columns', pad=15)
    axes[0].set_ylabel("Protein ratios [log2]\nto pseudo reference")
    axes[1].set_ylabel("Total protein intensity")
    fig.tight_layout()
    return fig, axes


def sample_pca(
    qtable: Qtable,
    tag: str = "Intensity",
    pc_x: str = "PC1",
    pc_y: str = "PC2",
    remove_invalid: bool = True,
) -> (plt.Figure, plt.Axes):
    """Figure to compare sample similarities with PCA.

    PCA of log2 transformed, mean centered intensity values.

    Args:
        qtable: msreport.qtable.Qtable instance, which data is used for
            plotting.
        tag: String used for matching the intensity columns.
        pc_x: Principle component to plot on x-axis of the scatter plot.
        pc_y: Principle component to plot on y-axis of the scatter plot.
        remove_invalid: If true and the column 'Valid' is present, rows are
            filtered according to the boolean entries of 'Valid'.

    Returns:
        A matplotlib Figure and Axes object containing the PCA plots.
    """

    matrix = qtable.make_sample_matrix(tag, samples_as_columns=True)
    if remove_invalid and "Valid" in qtable.data:
        matrix = matrix[qtable.data["Valid"]]

    matrix = matrix.replace({0: np.nan})
    matrix = matrix[np.isfinite(matrix).sum(axis=1) > 0]
    if not msreport.helper.intensities_in_logspace(matrix):
        matrix = np.log2(matrix)
    matrix[matrix.isna()] = 0

    matrix = matrix.transpose()
    sample_index = matrix.index.tolist()
    matrix = sklearn.preprocessing.scale(matrix, with_std=False)

    n_components = min(len(sample_index), 9)
    pca = sklearn.decomposition.PCA(n_components=n_components)
    components = pca.fit_transform(matrix)
    component_labels = ["PC{}".format(i + 1) for i in range(components.shape[1])]
    components_table = pd.DataFrame(
        data=components, columns=component_labels, index=sample_index
    )
    variance = pca.explained_variance_ratio_ * 100
    variance_lookup = dict(zip(component_labels, variance))

    # Prepare colors
    color_wheel = ColorWheelDict()
    experiments = qtable.get_experiments()
    _ = [color_wheel[exp] for exp in experiments]

    # Prepare figure
    num_legend_cols = 3
    legendheight = 0.2 + 0.2 * np.ceil(len(experiments) / num_legend_cols)
    plotheight = 3.7
    figheight = plotheight + legendheight
    figwidth = 4.3 + n_components * 0.2
    width_ratios = [4, 0.2 + n_components * 0.25]
    figsize = (figwidth, figheight)

    sns.set_style("white")
    fig, axes = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": width_ratios}
    )

    # Comparison of two principle components
    ax = axes[0]
    for sample, data in components_table.iterrows():
        experiment = qtable.get_experiment(sample)
        label = sample.replace(experiment, "").strip().strip("_")
        color = color_wheel[experiment]
        ax.scatter(
            data[pc_x],
            data[pc_y],
            color=color,
            edgecolor="#999999",
            lw=1,
            s=50,
            label=experiment,
        )
        ax.annotate(label, (data[pc_x], data[pc_y]))
    ax.tick_params(axis="both", labelsize=9)
    ax.set_xlabel(f"{pc_x} ({variance_lookup[pc_x]:.2f}%)", size=12)
    ax.set_ylabel(f"{pc_y} ({variance_lookup[pc_y]:.2f}%)", size=12)
    ax.grid(axis="both", linestyle="dotted", linewidth=1)

    # Explained variance bar plot
    ax = axes[1]
    xpos = range(len(variance))
    ax.bar(xpos, variance, color="#D0D0D0", edgecolor="#000000")
    ax.set_xticks(xpos)
    ax.set_xticklabels(component_labels, rotation="vertical", ha="center")
    ax.tick_params(axis="both", labelsize=9)
    ax.set_ylabel("Explained variance", size=12)
    ax.grid(axis="y", linestyle="dashed", linewidth=1)

    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    handles, labels = by_label.values(), by_label.keys()
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, 0.0),
        ncol=num_legend_cols,
        fontsize=9,
        loc="lower center",
    )
    legend_space = legendheight / figheight
    fig.suptitle(f'PCA of "{tag}" columns')
    fig.tight_layout(rect=[0, legend_space, 1, 1])

    return fig, axes


def box_and_bars(
    box_values: Iterable[Iterable[float]],
    bar_values: Iterable[float],
    group_names: list[str],
    colors: Optional[list[str]] = None,
) -> (plt.Figure, plt.Axes):
    """Generates a figure with horizontally aligned box and bar subplots.

    In the top subplot the box_values are displayed as box plots, in lower
    subplot the bar_values are displayed as bar plots. The figure width is
    automatically adjusted to the number of groups that will be plotted.
    The length of group_names must be the same as the length of the of the
    bar_values and the number of iterables from box_values. Each group from
    box_values and bar_values is horizontally aligned between the two subplots.

    Args:
        box_values: A sequence of sequences that each contain y values for
            generating a box plot.
        bar_values: A sequence of y values for generating bar plots.
        group_names: Used to label groups from box and bar plots.
        colors: Sequence of hex color codes for each group that is used for the
            boxes of the box and bar plots. Must be the same length as group
            names. If colors is None, boxes are colored in light grey.

    Returns:
        A matplotlib Figure and Axes object containing the box and bar plots.
    """
    assert len(box_values) == len(bar_values) == len(group_names)
    assert colors is None or len(colors) == len(group_names)
    if colors is None:
        colors = ["#D0D0D0" for _ in group_names]

    num_samples = len(group_names)
    x_values = range(num_samples)
    width = 0.8
    xlim = (-1 + 0.15, num_samples - 0.15)
    figwidth = (num_samples * 0.25) + 1.1
    figsize = (figwidth, 6)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, figsize=figsize, sharex=True)

    # Plot boxplots using the box_values
    ax = axes[0]
    ax.plot(xlim, (0, 0), color="#999999", lw=1, zorder=2)
    boxplots = ax.boxplot(
        box_values,
        positions=x_values,
        vert=True,
        showfliers=False,
        patch_artist=True,
        widths=width,
        medianprops={"color": "#000000"},
    )
    for color, box in zip(colors, boxplots["boxes"]):
        box.set(facecolor=color)
    ylim = ax.get_ylim()
    ax.set_ylim(min(-0.4, ylim[0]), max(0.401, ylim[1]))

    # Plot barplots using the bar_values
    ax = axes[1]
    ax.bar(x_values, bar_values, width=width, color=colors, edgecolor="#000000")
    ax.set_xticklabels(group_names, rotation=90)
    for ax_pos, ax in enumerate(axes):
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#000000")
            ax.spines[spine].set_linewidth(1)
        ax.grid(False, axis="x")
        ax.grid(axis="y", linestyle="dashed", linewidth=1, color="#cccccc")
    sns.despine(top=True, right=True)

    ax.set_xlim(xlim)
    fig.tight_layout()
    return fig, axes


def volcano_ma(qtable) -> list[(plt.Figure, plt.Axes)]:
    """Returns volcano and ma figure for each comparison group."""
    comparison_tag = " vs "
    data = qtable.data.copy()
    if "Valid" in qtable.data:
        data = data[qtable.data["Valid"]]

    experiments = qtable.get_experiments()

    possible_comparisons = itertools.permutations(experiments, 2)
    comparison_groups = []
    for i, j in possible_comparisons:
        comparison_group = comparison_tag.join([i, j])
        columns = msreport.helper.find_columns(data, comparison_group)
        if columns:
            comparison_groups.append(comparison_group)

    for variable in ["P-value", "Adjusted p-value"]:
        for column in msreport.helper.find_sample_columns(
            data, variable, comparison_groups
        ):
            data[column] = np.log10(data[column]) * -1

    scatter_size = 2 / (max(min(data.shape[0], 10000), 1000) / 1000)

    figures = []
    for comparison_group in comparison_groups:
        fig, axes = plt.subplots(1, 2, figsize=[8, 4], sharex=True)
        fig.suptitle(comparison_group)

        for ax, x_variable, y_variable in [
            (axes[0], "logFC", "P-value"),
            (axes[1], "logFC", "Average expression"),
        ]:
            x_col = " ".join([x_variable, comparison_group])
            y_col = " ".join([y_variable, comparison_group])
            x_values = data[x_col]
            y_values = data[y_col]
            ax.grid(axis="both", linestyle="dotted", linewidth=1)
            ax.scatter(x_values, y_values, s=scatter_size, color="#606060", zorder=3)
            ax.set_xlabel(x_variable)
            ax.set_ylabel(y_variable)

        fig.tight_layout()
        figures.append((fig, axes))
    return figures


def contaminants(
    qtable: Qtable, ibaq_tag: str = "iBAQ intensity"
) -> (plt.Figure, plt.Axes):
    """Returns a barplot of relative contaminant amounts per sample."""
    data = qtable.make_sample_matrix(ibaq_tag, samples_as_columns=True)
    ibaq_sums = data.sum()
    ribaq = data / ibaq_sums * 100
    contaminants = qtable.data["Potential contaminant"]
    sample_names = data.columns.to_list()

    x_values = range(ribaq.shape[1])
    bar_values = ribaq[contaminants].sum(axis=0)
    width = 0.8
    colors = "#C0C0C0"

    fig, ax = plt.subplots()
    ax.bar(
        x_values, bar_values, width=width, color=colors, edgecolor="#000000", zorder=3
    )
    ax.set_xticks(x_values)
    ax.set_xticklabels(sample_names, rotation=90)
    ax.set_ylabel("Sum riBAQ [%]")

    ax.set_ylim(0, max(5, ax.get_ylim()[1]))
    sns.despine(top=True, right=True)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#000000")
        ax.spines[spine].set_linewidth(1)
    ax.grid(False, axis="x")
    ax.grid(axis="y", linestyle="dashed", linewidth=1, color="#cccccc")

    fig.suptitle("Relative amount of contaminants")

    fig.tight_layout()
    return fig, ax


def expression_comparison(
    qtable: Qtable,
    group: list[str, str],
    comparison_tag: str = " vs ",
    optional: bool = False,
) -> (plt.Figure, plt.Axes):
    exp_1, exp_2 = group
    comparison_group = "".join([exp_1, comparison_tag, exp_2])

    data = qtable.data.copy()
    if "Valid" in qtable.data:
        data = data[qtable.data["Valid"]]

    mask = (data[f"Events {exp_1}"] + data[f"Events {exp_2}"]) > 0
    data = data[mask]

    only_exp_1 = data[f"Events {exp_2}"] == 0
    only_exp_2 = data[f"Events {exp_1}"] == 0
    mask_both = np.invert(np.any([only_exp_1, only_exp_2], axis=0))

    # Test if plotting maximum intensity is better than average
    if optional:
        max_values = np.max(
            [data[f"Expression {exp_2}"], data[f"Expression {exp_1}"]], axis=0
        )
        data[f"Average expression {comparison_group}"] = max_values

    def scattersize(df: pd.DataFrame) -> float:
        return min(max(np.sqrt(scatter_area / df.shape[0]), 0.5), 4)

    scatter_area = 5000

    width_ratios = [1, 5, 1]
    fig, axes = plt.subplots(
        1, 3, figsize=[6, 4], sharey=True, gridspec_kw={"width_ratios": width_ratios}
    )

    for ax, mask, exp in [(axes[2], only_exp_1, exp_1), (axes[0], only_exp_2, exp_2)]:
        values = data[mask]
        s = scattersize(values)
        y_variable = f"Expression {exp}"
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UserWarning)
            try:
                sns.swarmplot(
                    y=values[y_variable],
                    size=np.sqrt(s * 2),
                    marker="o",
                    alpha=0.75,
                    color="#606060",
                    edgecolor="none",
                    ax=ax,
                )
            except UserWarning:
                ax.cla()
                sns.stripplot(
                    y=values[y_variable],
                    jitter=True,
                    size=np.sqrt(s * 2),
                    marker="o",
                    alpha=0.75,
                    color="#606060",
                    edgecolor="none",
                    ax=ax,
                )
                ax.set_xlim(-0.2, 0.2)
        ax.grid(axis="y", linestyle="dotted", linewidth=1)
        ax.set_ylabel(f"Average expression {exp}")
    axes[0].set_title(f"Absent in\n{exp_1}", fontsize=9)
    axes[2].set_title(f"Absent in\n{exp_2}", fontsize=9)

    ax = axes[1]
    values = data[mask_both]
    s = scattersize(values)
    x_variable = f"logFC"
    y_variable = f"Average expression"
    x_col = " ".join([x_variable, comparison_group])
    y_col = " ".join([y_variable, comparison_group])
    x_values = values[x_col]
    y_values = values[y_col]
    ax.grid(axis="both", linestyle="dotted", linewidth=1)
    ax.scatter(x_values, y_values, s=s, alpha=0.75, color="#606060", zorder=3)
    ax.set_xlabel(x_variable, fontsize=9)
    ax.set_title(comparison_group, fontsize=12)
    if optional:
        ax.set_ylabel("Maximum expression")

    fig.tight_layout()
    return fig, axes
