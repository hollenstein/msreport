import itertools
import warnings

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import msreport.helper
from msreport.qtable import Qtable

from ._partial_plots import box_and_bars
from .style import ColorWheelDict, with_active_style


@with_active_style
def missing_values_vertical(
    qtable: Qtable,
    exclude_invalid: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Vertical bar plot to analyze the completeness of quantification.

    Requires the columns "Missing experiment_name" and "Events experiment_name", which
    are added by calling msreport.analyze.analyze_missingness(qtable: Qtable).

    Args:
        qtable: A `Qtable` instance, which data is used for plotting.
        exclude_invalid: If True, rows are filtered according to the Boolean entries of
            the "Valid" column.

    Returns:
        A matplotlib Figure and a list of Axes objects containing the missing values
        plots.
    """
    # add a deprecation warning here
    warnings.warn(
        (
            "The function `missing_values_vertical` is deprecated. Use"
            "`missing_values_horizontal` instead."
        ),
        DeprecationWarning,
        stacklevel=2,
    )

    experiments = qtable.get_experiments()
    num_experiments = len(experiments)
    qtable_data = qtable.get_data(exclude_invalid=exclude_invalid)

    barwidth = 0.8
    barcolors = ["#31A590", "#FAB74E", "#EB3952"]
    figwidth = (num_experiments * 1.2) + 0.5
    figsize = (figwidth, 3.5)
    xtick_labels = ["No missing", "Some missing", "All missing"]

    fig, axes = plt.subplots(1, num_experiments, figsize=figsize, sharey=True)
    for exp_num, exp in enumerate(experiments):
        ax = axes[exp_num]

        exp_missing = qtable_data[f"Missing {exp}"]
        exp_values = qtable_data[f"Events {exp}"]
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
        ax.grid(False, axis="x")
    sns.despine(top=True, right=True)
    fig.tight_layout()
    return fig, axes


@with_active_style
def missing_values_horizontal(
    qtable: Qtable,
    exclude_invalid: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Horizontal bar plot to analyze the completeness of quantification.

    Requires the columns "Missing experiment_name" and "Events experiment_name", which
    are added by calling msreport.analyze.analyze_missingness(qtable: Qtable).

    Args:
        qtable: A `Qtable` instance, which data is used for plotting.
        exclude_invalid: If True, rows are filtered according to the Boolean entries of
            the "Valid" column.

    Returns:
        A matplotlib Figure and Axes object, containing the missing values plot.
    """
    experiments = qtable.get_experiments()
    num_experiments = len(experiments)
    qtable_data = qtable.get_data(exclude_invalid=exclude_invalid)

    data: dict[str, list] = {"exp": [], "max": [], "some": [], "min": []}
    for exp in experiments:
        exp_missing = qtable_data[f"Missing {exp}"]
        total = len(exp_missing)
        num_replicates = len(qtable.get_samples(exp))
        missing_all = (exp_missing == num_replicates).sum()
        missing_none = (exp_missing == 0).sum()
        with_missing_some = total - missing_all

        data["exp"].append(exp)
        data["max"].append(total)
        data["some"].append(with_missing_some)
        data["min"].append(missing_none)

    bar_width = 0.35

    suptitle_space_inch = 0.4
    ax_height_inch = num_experiments * bar_width
    ax_width_inch = 4
    fig_height = ax_height_inch + suptitle_space_inch
    fig_width = ax_width_inch
    fig_size = (fig_width, fig_height)

    subplot_top = 1 - (suptitle_space_inch / fig_height)

    fig, ax = plt.subplots(figsize=fig_size)
    fig.subplots_adjust(bottom=0, top=subplot_top, left=0, right=1)
    fig.suptitle("Completeness of quantification per experiment", y=1)

    sns.barplot(y="exp", x="max", data=data, label="All missing", color="#EB3952")
    sns.barplot(y="exp", x="some", data=data, label="Some missing", color="#FAB74E")
    sns.barplot(y="exp", x="min", data=data, label="None missing", color="#31A590")

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_xlim(0, total)

    ax.legend().remove()
    handles, labels = ax.get_legend_handles_labels()
    legend_ygap_inches = 0.3
    legend_bbox_y = 0 - (legend_ygap_inches / fig.get_figheight())

    fig.legend(
        handles[::-1],
        labels[::-1],
        bbox_to_anchor=(0.5, legend_bbox_y),
        loc="upper center",
        ncol=3,
        frameon=False,
        borderaxespad=0,
        handlelength=0.95,
        handleheight=1,
    )

    ax.tick_params(axis="y", labelsize=plt.rcParams["axes.labelsize"])
    ax.grid(axis="x", linestyle="solid")
    sns.despine(fig=fig, top=True, right=True, bottom=True)

    return fig, ax


@with_active_style
def contaminants(
    qtable: Qtable, tag: str = "iBAQ intensity"
) -> tuple[plt.Figure, plt.Axes]:
    """A bar plot that displays relative contaminant amounts (iBAQ) per sample.

    Requires "iBAQ intensity" columns for each sample, and a "Potential contaminant"
    column to identify the potential contaminant entries.

    The relative iBAQ values are calculated as:
    sum of contaminant iBAQ intensities / sum of all iBAQ intensities * 100

    It is possible to use intensity columns that are either log-transformed or not. The
    intensity values undergo an automatic evaluation to determine if they are already
    in log-space, and if necessary, they are transformed accordingly.

    Args:
        qtable: A `Qtable` instance, which data is used for plotting.
        tag: A string that is used to extract iBAQ intensity containing columns.
            Default "iBAQ intensity".

    Raises:
        ValueError: If the "Potential contaminant" column is missing in the Qtable data.
            If the Qtable does not contain any columns for the specified 'tag'.

    Returns:
        A matplotlib Figure and an Axes object, containing the contaminants plot.
    """
    if "Potential contaminant" not in qtable.data.columns:
        raise ValueError(
            "The 'Potential contaminant' column is missing in the Qtable data."
        )
    data = qtable.make_sample_table(tag, samples_as_columns=True)
    if data.empty:
        raise ValueError(f"The Qtable does not contain any '{tag}' columns.")
    if msreport.helper.intensities_in_logspace(data):
        data = np.power(2, data)

    relative_intensity = data / data.sum() * 100
    contaminants = qtable["Potential contaminant"]
    samples = data.columns.to_list()

    color_wheel = ColorWheelDict()
    colors = [color_wheel[exp] for exp in qtable.get_experiments(samples)]
    dark_colors = [
        color_wheel.modified_color(exp, 0.4) for exp in qtable.get_experiments(samples)
    ]

    num_samples = len(samples)
    x_values = range(relative_intensity.shape[1])
    bar_values = relative_intensity[contaminants].sum(axis=0)

    suptitle_space_inch = 0.4
    ax_height_inch = 1.6
    bar_width_inches = 0.24
    x_padding = 0.24

    fig_height = ax_height_inch + suptitle_space_inch
    fig_width = (num_samples + (2 * x_padding)) * bar_width_inches
    fig_size = (fig_width, fig_height)

    subplot_top = 1 - (suptitle_space_inch / fig_height)

    bar_width = 0.8
    bar_half_width = 0.5
    lower_xbound = (0 - bar_half_width) - x_padding
    upper_xbound = (num_samples - 1) + bar_half_width + x_padding
    min_upper_ybound = 5

    fig, ax = plt.subplots(figsize=fig_size)
    fig.subplots_adjust(bottom=0, top=subplot_top, left=0, right=1)
    fig.suptitle("Relative amount of contaminants", y=1)

    ax.bar(
        x_values,
        bar_values,
        width=bar_width,
        color=colors,
        edgecolor=dark_colors,
        zorder=3,
    )
    ax.set_xticks(x_values)
    ax.set_xticklabels(samples, fontsize=plt.rcParams["axes.labelsize"], rotation=90)
    ax.set_ylabel(f"Sum contaminant\n{tag} [%]")

    ax.grid(False, axis="x")
    sns.despine(top=True, right=True)

    ax.set_ylim(0, max(min_upper_ybound, ax.get_ylim()[1]))
    ax.set_xlim(lower_xbound, upper_xbound)
    return fig, ax


@with_active_style
def sample_intensities(
    qtable: Qtable, tag: str = "Intensity", exclude_invalid: bool = True
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Figure to compare the overall quantitative similarity of samples.

    Generates two subplots to compare the intensities of multiple samples. For the top
    subplot a pseudo reference sample is generated by calculating the average intensity
    values of all samples. For each row and sample the log2 ratios to the pseudo
    reference are calculated. Only rows without missing values are selected, and for
    each sample the log2 ratios to the pseudo reference are displayed as a box plot. The
    lower subplot displays the summed intensity of all rows per sample as bar plots.

    It is possible to use intensity columns that are either log-transformed or not. The
    intensity values undergo an automatic evaluation to determine if they are already
    in log-space, and if necessary, they are transformed accordingly.

    Args:
        qtable: A `Qtable` instance, which data is used for plotting.
        tag: A string that is used to extract intensity containing columns.
            Default "Intensity".
        exclude_invalid: If True, rows are filtered according to the Boolean entries of
            the "Valid" column.

    Returns:
        A matplotlib Figure and a list of Axes objects, containing the intensity plots.
    """
    table = qtable.make_sample_table(
        tag, samples_as_columns=True, exclude_invalid=exclude_invalid
    )

    table = table.replace({0: np.nan})
    if msreport.helper.intensities_in_logspace(table):
        log2_table = table
        table = np.power(2, log2_table)
    else:
        log2_table = np.log2(table)
    samples = table.columns.tolist()

    finite_values = log2_table.isna().sum(axis=1) == 0
    pseudo_ref = np.nanmean(log2_table[finite_values], axis=1)
    log2_ratios = log2_table[finite_values].subtract(pseudo_ref, axis=0)

    bar_values = table.sum()
    box_values = [log2_ratios[c] for c in log2_ratios.columns]
    color_wheel = ColorWheelDict()
    colors = [color_wheel[exp] for exp in qtable.get_experiments(samples)]
    edge_colors = [
        color_wheel.modified_color(exp, 0.4) for exp in qtable.get_experiments(samples)
    ]

    fig, axes = box_and_bars(
        box_values, bar_values, samples, colors=colors, edge_colors=edge_colors
    )
    fig.suptitle(f'Comparison of "{tag}" values', y=1)
    axes[0].set_ylabel("Ratio [log2]\nto pseudo reference")
    axes[1].set_ylabel("Total intensity")
    return fig, axes


@with_active_style
def sample_correlation(
    qtable, exclude_invalid: bool = True, labels: bool = False
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Generates a pair-wise correlation matrix of samples 'Expression' values.

    Correlation values are calculated using the Pearson method and the "Expression"
    values.

    Args:
        qtable: A `Qtable` instance, which data is used for plotting.
        exclude_invalid: If True, rows are filtered according to the Boolean entries of
            the "Valid" column.
        labels: If True, correlation values are displayed in the heatmap.

    Raises:
        ValueError: If less than two samples are present in the qtable.

    Returns:
        A matplotlib Figure and a list of Axes objects, containing the correlation
        matrix plot and the color bar
    """
    num_samples = qtable.design.shape[0]
    if num_samples < 2:
        raise ValueError(
            "At least two samples are required to generate a correlation matrix."
        )
    data = qtable.make_expression_table(
        samples_as_columns=True, exclude_invalid=exclude_invalid
    )
    samples = data.columns.tolist()
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    num_cells = num_samples - 1
    cell_size_inch = 0.3
    suptitle_space_inch = 0.4
    ax_height_inch = ax_width_inch = cell_size_inch * num_cells
    ax_wspace_inch = 0.4
    cbar_height_inch = max(1.2, min(3, cell_size_inch * num_cells))
    cbar_width_inch = 0.27
    width_ratios = [ax_width_inch, cbar_width_inch]
    subplot_wspace = ax_wspace_inch / np.mean([ax_width_inch, cbar_width_inch])

    fig_width = ax_width_inch + cbar_width_inch + ax_wspace_inch
    fig_height = ax_height_inch + suptitle_space_inch
    fig_size = (fig_width, fig_height)

    subplot_top = 1 - (suptitle_space_inch / fig_height)
    cbar_width = cbar_width_inch / fig_width
    cbar_height = cbar_height_inch / fig_height
    cbar_x0 = (ax_width_inch + ax_wspace_inch) / fig_width
    cbar_y0 = (ax_height_inch / fig_height) - cbar_height

    fig, axes = plt.subplots(
        1,
        2,
        figsize=fig_size,
        gridspec_kw={
            "bottom": 0,
            "top": subplot_top,
            "left": 0,
            "right": 1,
            "wspace": subplot_wspace,
            "width_ratios": width_ratios,
        },
    )
    fig.suptitle('Pairwise correlation matrix of sample "Expression" values', y=1)
    ax_heatmap, ax_cbar = axes
    ax_cbar.set_position((cbar_x0, cbar_y0, cbar_width, cbar_height))

    palette = sns.color_palette("rainbow", desat=0.8)
    cmap = mcolors.LinearSegmentedColormap.from_list("rainbow_desat", palette)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=0.5,
        square=False,
        linewidths=0.5,
        ax=ax_heatmap,
    )
    cbar = ax_heatmap.collections[0].colorbar
    if cbar is not None:
        cbar.remove()
    fig.colorbar(ax_heatmap.collections[0], cax=ax_cbar)

    if labels:
        for i, j in itertools.product(range(num_cells + 1), range(num_cells + 1)):
            if i <= j:
                continue
            corr_value = corr.iloc[i, j]
            ax_heatmap.text(
                j + 0.5,
                i + 0.5,
                f"{corr_value:.2f}",
                ha="center",
                va="center",
                fontsize=8,  # Fontsize cannot be larger to fit in the cell
            )
    # Need to manually set ticks because sometimes not all are properly included
    ax_heatmap.set_yticks([i + 0.5 for i in range(1, len(samples))])
    ax_heatmap.set_yticklabels(samples[1:], rotation=0)
    ax_heatmap.set_xticks([i + 0.5 for i in range(0, len(samples) - 1)])
    ax_heatmap.set_xticklabels(samples[:-1], rotation=90)

    ax_heatmap.grid(False)
    ax_heatmap.tick_params(labelsize=plt.rcParams["axes.labelsize"])
    ax_heatmap.set_xlim(0, num_cells)
    ax_heatmap.set_ylim(1 + num_cells, 1)

    sns.despine(left=False, bottom=False, ax=ax_heatmap)
    for ax in [ax_heatmap, ax_cbar]:
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_linewidth(0.75)
    return fig, axes
