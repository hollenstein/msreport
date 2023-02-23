from collections.abc import Iterable
from collections import UserDict
import itertools
from typing import Optional
import re
import warnings

import adjustText
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.preprocessing
import sklearn.decomposition

from msreport.qtable import Qtable
import msreport.helper


def set_dpi(dpi: int) -> None:
    """Changes the default dots per inch settings for matplotlib plots.

    This effectively makes figures smaller or larger, without affecting the relative
    sizes of elements within the figures.

    Args:
        dpi: New default dots per inch.
    """
    plt.rcParams["figure.dpi"] = dpi


class ColorWheelDict(UserDict):
    """Lookup dictionary that maps keys to hex colors by using a color wheel.

    When a key is not present the first color of the color wheel is added as the value,
    and the color is moved from the beginning to the end of the color wheel. If no list
    of colors is specified, a default list of ten colors is added to the color wheel.
    It is also possible to manually set key and color pairs by using the same syntax as
    for a regular dictionary.
    """

    def __init__(self, colors: Optional[list[str]] = None):
        """Initializes a ColorWheelDict.

        Args:
            colors: Optional, a list of hex colors used for the color wheel. By default
                a list with ten colors is used.
        """
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
    exclude_invalid: bool = True,
) -> (plt.Figure, list[plt.Axes]):
    """Vertical bar plot to analyze the completeness of quantification.

    Requires the columns "Missing experiment_name" and "Events experiment_name", which
    are added by calling msreport.analyze.analyze_missingness(qtable: Qtable).

    Args:
        qtable: A Qtable instance, which data is used for plotting.
        exclude_invalid: Optional, if True rows are filtered according to the Boolean
            entries of "Valid"; default True.
    Returns:
        A matplotlib Figure and a list of Axes objects containing the missing values
        plots.
    """
    experiments = qtable.get_experiments()
    num_experiments = len(experiments)
    qtable_data = qtable.get_data(exclude_invalid=exclude_invalid)

    barwidth = 0.8
    barcolors = ["#31A590", "#FAB74E", "#EB3952"]
    figwidth = (num_experiments * 1.2) + 0.5
    figsize = (figwidth, 3.5)
    xtick_labels = ["No missing", "Some missing", "All missing"]

    sns.set_style("whitegrid")
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
    exclude_invalid: bool = True,
) -> (plt.Figure, plt.Axes):
    """Horizontal bar plot to analyze the completeness of quantification.

    Requires the columns "Missing experiment_name" and "Events experiment_name", which
    are added by calling msreport.analyze.analyze_missingness(qtable: Qtable).

    Args:
        qtable: A Qtable instance, which data is used for plotting.
        exclude_invalid: Optional, if True rows are filtered according to the Boolean
            entries of "Valid"; default True.
    Returns:
        A matplotlib Figure and Axes object, containing the missing values plot.
    """
    experiments = qtable.get_experiments()
    num_experiments = len(experiments)
    qtable_data = qtable.get_data(exclude_invalid=exclude_invalid)

    data = {"exp": [], "max": [], "some": [], "min": []}
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


def contaminants(qtable: Qtable, tag: str = "iBAQ intensity") -> (plt.Figure, plt.Axes):
    """A bar plot that displays relative contaminant amounts (iBAQ) per sample.

    The relative iBAQ values are calculated as:
    sum of contaminant iBAQ intensities / sum of all iBAQ intensities * 100

    Requires "iBAQ intensity" columns for each sample, and a "Potential contaminant"
    column to identify the potential contaminant entries.

    Args:
        qtable: A Qtable instance, which data is used for plotting.
        tag: String used for matching the iBAQ intensity columns; default
            "iBAQ intensity", which corresponds to the MsReport convention.

    Returns:
        A matplotlib Figure and Axes object, containing the contaminants plot.
    """
    data = qtable.make_sample_table(tag, samples_as_columns=True)
    relative_intensity = data / data.sum() * 100
    contaminants = qtable["Potential contaminant"]
    samples = data.columns.to_list()
    num_samples = len(samples)

    x_values = range(relative_intensity.shape[1])
    bar_values = relative_intensity[contaminants].sum(axis=0)

    color_wheel = ColorWheelDict()
    colors = [color_wheel[exp] for exp in qtable.get_experiments(samples)]
    width = 0.8
    xlim_pad = 0.5
    xlim = ((width / 2 + xlim_pad) * -1, (num_samples - 1 + width / 2 + xlim_pad))
    min_upper_ylim = 5
    figwidth = (num_samples * 0.25) + 1.05
    figsize = (figwidth, 3)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        x_values, bar_values, width=width, color=colors, edgecolor="#000000", zorder=3
    )
    ax.set_xticks(x_values)
    ax.set_xticklabels(samples, rotation=90)
    ax.set_ylabel(f"Sum relative\n{tag} [%]")

    ax.set_ylim(0, max(min_upper_ylim, ax.get_ylim()[1]))
    ax.set_xlim(xlim)
    sns.despine(top=True, right=True)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#000000")
        ax.spines[spine].set_linewidth(1)
    ax.grid(False, axis="x")
    ax.grid(axis="y", linestyle="dashed", linewidth=1, color="#cccccc")

    ax.set_title("Relative amount of contaminants")

    fig.tight_layout()
    return fig, ax


def sample_intensities(
    qtable: Qtable, tag: str = "Intensity", exclude_invalid: bool = True
) -> (plt.Figure, list[plt.Axes]):
    """Figure to compare the overall quantitative similarity of samples.

    Generates two subplots to compare the intensities of multiple samples. For the top
    subplot a pseudo reference sample is generated by calculating the average intensity
    values of all samples. For each row and sample the log2 ratios to the pseudo
    reference are calculated. Only rows without missing values are selected, and for
    each sample the log2 ratios to the pseudo reference are displayed as a box plot. The
    lower subplot displays the summed intensity of all rows per sample as bar plots.

    Args:
        qtable: A Qtable instance, which data is used for plotting.
        tag: String used for matching the intensity columns.
        exclude_invalid: Optional, if True rows are filtered according to the Boolean
            entries of "Valid"; default True.

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
    fig, axes = box_and_bars(box_values, bar_values, samples, colors=colors)
    axes[0].set_title(f'Comparison of "{tag}" columns', pad=15)
    axes[0].set_ylabel("Protein ratios [log2]\nto pseudo reference")
    axes[1].set_ylabel("Total protein intensity")
    fig.tight_layout()
    return fig, axes


def replicate_ratios(
    qtable: Qtable,
    exclude_invalid: bool = True,
    xlim: Iterable[float] = (-2, 2),
) -> (plt.Figure, list[plt.Axes]):
    """Figure to compare the similarity of expression values between replicates.

    Intended to evaluate the bulk distribution of expression values within experiments,
    after normalization. Plots from one experiment are placed in the same row. For each
    experiment, samples are compared pair-wise and for each sample comparison the
    distribution of the log2 ratios is shown as a density plot.

    Requires log2 transformed expression values.

    Args:
        qtable: A Qtable instance, which data is used for plotting.
        exclude_invalid: Optional, if True rows are filtered according to the Boolean
            entries of "Valid"; default True.
        xlim: Optional, specifies the displayed log2 ratio range on the x-axis. Default
            is from -2 to 2.

    Returns:
        A matplotlib Figure and a list of Axes objects, containing the comparison plots.
    """
    tag: str = "Expression"
    table = qtable.make_sample_table(
        tag, samples_as_columns=True, exclude_invalid=exclude_invalid
    )
    design = qtable.get_design()

    experiments = design["Experiment"].unique().tolist()
    num_experiments = len(experiments)
    max_replicates = max([len(qtable.get_samples(exp)) for exp in experiments])
    max_combinations = len(list(itertools.combinations(range(max_replicates), 2)))

    figheight = (num_experiments * 0.85) + 0.64
    figwidth = (max_combinations * 1.5) + 0.75
    figsize = (figwidth, figheight)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(
        num_experiments, max_combinations, figsize=figsize, sharex=True
    )

    color_wheel = ColorWheelDict()
    _ = [color_wheel[exp] for exp in experiments]

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
            ax.set_title(title, fontsize=10)
            ax.set_yticklabels("")
            ax.set_ylabel(ylabel, rotation=90, fontsize=10, va="center")
            ax.set_xlabel("")
            ax.tick_params(axis="both", labelsize=8)
            ax.locator_params(axis="x", nbins=5)

    axes[0, 0].set_xlim(xlim)
    for ax in axes.flatten():
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#000000")
            ax.spines[spine].set_linewidth(0.5)
        ax.plot((0, 0), ax.get_ylim(), color="#999999", lw=1, zorder=2)
        ax.grid(False, axis="y")
        ax.grid(axis="x", linestyle="dashed", linewidth=1, color="#cccccc")
    sns.despine(top=True, right=True)
    fig.suptitle("Protein ratios [log2] between replicates", fontsize=10)
    fig.tight_layout()

    return fig, axes


def experiment_ratios(
    qtable: Qtable,
    experiments: Optional[str] = None,
    exclude_invalid: bool = True,
    ylim: Iterable[float] = (-2, 2),
) -> (plt.Figure, list[plt.Axes]):
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
        qtable: A Qtable instance, which data is used for plotting.
        experiments: Optional, list of experiments that will be displayed. If None, all
            experiments from `qtable.design` will be used.
        exclude_invalid: Optional, if True rows are filtered according to the Boolean
            entries of "Valid"; default True.
        ylim: Optional, specifies the displayed log2 ratio range on the y-axis. Default
            is from -2 to 2.

    Returns:
        A matplotlib Figure and a list of Axes objects, containing the comparison plots.
    """
    tag: str = "Expression"
    qtable_data = qtable.get_data(exclude_invalid=exclude_invalid)
    if experiments is None:
        experiments = qtable.design["Experiment"].unique().tolist()

    column_mapping = {f"{tag} {exp}": exp for exp in experiments}
    exp_data = qtable_data[column_mapping.keys()]
    exp_data = exp_data.rename(columns=column_mapping)

    quant_mask = np.all(
        [(qtable_data[f"Events {exp}"] > 0) for exp in experiments], axis=0
    )
    exp_data = exp_data[quant_mask]
    pseudo_ref = np.nanmean(exp_data, axis=1)
    exp_ratios = exp_data.subtract(pseudo_ref, axis=0)

    color_wheel = ColorWheelDict()
    num_experiments = len(experiments)
    figwidth = (num_experiments * 0.75) + 0.82
    figheight = 2.5
    figsize = (figwidth, figheight)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, num_experiments, figsize=figsize, sharey=True)

    for exp_pos, experiment in enumerate(experiments):
        ax = axes[exp_pos]
        values = exp_ratios[experiment]
        color = color_wheel[experiment]
        sns.kdeplot(y=values, fill=True, ax=ax, zorder=3, color=color, alpha=0.5)
        if exp_pos == 0:
            ax.text(
                x=ax.get_xlim()[1] / 20,
                y=ylim[1] * 0.95,
                s=f"n={str(len(values))}",
                va="top",
                ha="left",
                fontsize=8,
            )
        ax.set_xticklabels("")
        ax.tick_params(axis="both", labelsize=8)
        ax.set_xlabel(experiment, rotation=90)

    axes[0].set_ylabel("Protein ratios [log2]\nto pseudo reference")
    axes[0].set_ylim(ylim)
    for ax_pos, ax in enumerate(axes):
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#000000")
            ax.spines[spine].set_linewidth(0.5)
        ax.plot(ax.get_xlim(), (0, 0), color="#999999", lw=1, zorder=2)
        ax.grid(False, axis="x")
        ax.grid(axis="y", linestyle="dashed", linewidth=1, color="#cccccc")
    sns.despine(top=True, right=True)
    fig.tight_layout()
    return fig, axes


def sample_pca(
    qtable: Qtable,
    tag: str = "Expression",
    pc_x: str = "PC1",
    pc_y: str = "PC2",
    exclude_invalid: bool = True,
) -> (plt.Figure, list[plt.Axes]):
    """Figure to compare sample similarities with a principle component analysis.

    On the left subplots two PCA components of log2 transformed, mean centered intensity
    values are shown. On the right subplot the explained variance of the principle
    components is display as barplots.

    Args:
        qtable: A Qtable instance, which data is used for the PCA analysis.
        tag: String used for matching the intensity columns, default "Expression".
            Intensity values are evaluated whether they are likely to be in log-space,
            and if not are log2 transformed for the PCA analysis.
        pc_x: Principle component to plot on x-axis of the scatter plot, default "PC1".
            The number of calculated principal components is equal to the number of
            samples.
        pc_y: Principle component to plot on y-axis of the scatter plot, default "PC2".
            The number of calculated principal components is equal to the number of
            samples.
        exclude_invalid: Optional, if True rows are filtered according to the Boolean
            entries of "Valid"; default True.

    Returns:
        A matplotlib Figure and a list of Axes objects, containing the PCA plots.
    """
    table = qtable.make_sample_table(
        tag, samples_as_columns=True, exclude_invalid=exclude_invalid
    )
    design = qtable.get_design()

    table = table.replace({0: np.nan})
    table = table[np.isfinite(table).sum(axis=1) > 0]
    if not msreport.helper.intensities_in_logspace(table):
        table = np.log2(table)
    table[table.isna()] = 0

    table = table.transpose()
    sample_index = table.index.tolist()
    table = sklearn.preprocessing.scale(table, with_std=False)

    n_components = min(len(sample_index), 9)
    pca = sklearn.decomposition.PCA(n_components=n_components)
    components = pca.fit_transform(table)
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
    texts = []
    for sample, data in components_table.iterrows():
        experiment = qtable.get_experiment(sample)
        label = design.loc[(design["Sample"] == sample), "Replicate"].tolist()[0]
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
        texts.append(ax.text(data[pc_x], data[pc_y], label, fontdict={"fontsize": 9}))
    adjustText.adjust_text(
        texts,
        force_text=0.15,
        arrowprops=dict(arrowstyle="-", color="#ebae34", lw=0.5),
        lim=20,
        ax=ax,
    )
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


def volcano_ma(
    qtable: Qtable,
    experiment_pair: list[str],
    comparison_tag: str = " vs ",
    pvalue_tag: str = "P-value",
    special_proteins: Optional[list[str]] = None,
    exclude_invalid: bool = True,
) -> (plt.Figure, list[plt.Axes]):
    """Generates a volcano and an MA plot for the comparison of two experiments.

    Args:
        qtable: A Qtable instance, which data is used for plotting.
        experiment_pair: The names of the two experiments that will be compared,
            experiments must be present in qtable.design.
        comparison_tag: String used in comparison columns to separate a pair of
            experiments; default " vs ", which corresponds to the MsReport convention.
        pvalue_tag: String used for matching the pvalue columns; default "P-value",
            which corresponds to the MsReport convention.
        special_proteins: Optional, allows to specify a list of entries from the
            "Representative Protein" column to be annotated. Entries are annotated with
            values from the "Gene Name" column if present, otherwise from the
            "Representative Protein" column.
        exclude_invalid: Optional, if True rows are filtered according to the Boolean
            entries of "Valid"; default True.

    Returns:
        A matplotlib Figure object and a list of two Axes objects containing the volcano
        and the MA plot.
    """
    comparison_group = comparison_tag.join(experiment_pair)

    special_proteins = special_proteins if special_proteins is not None else []
    data = qtable.get_data(exclude_invalid=exclude_invalid)
    annotation_column = (
        "Gene name" if "Gene name" in data.columns else "Representative protein"
    )
    scatter_size = 2 / (max(min(data.shape[0], 10000), 1000) / 1000)

    masks = {
        "highlight": data["Representative protein"].isin(special_proteins),
        "default": ~data["Representative protein"].isin(special_proteins),
    }
    params = {
        "highlight": {
            "s": 10,
            "color": "#FAB74E",
            "edgecolor": "#000000",
            "lw": 0.2,
            "zorder": 3,
        },
        "default": {"s": scatter_size, "color": "#A0A0A0", "zorder": 2},
    }

    for column in msreport.helper.find_sample_columns(
        data, pvalue_tag, [comparison_group]
    ):
        data[column] = np.log10(data[column]) * -1

    fig, axes = plt.subplots(1, 2, figsize=[8, 4], sharex=True)
    fig.suptitle(comparison_group)

    for ax, x_variable, y_variable in [
        (axes[0], "Ratio [log2]", pvalue_tag),
        (axes[1], "Ratio [log2]", "Average expression"),
    ]:
        x_col = " ".join([x_variable, comparison_group])
        y_col = " ".join([y_variable, comparison_group])
        x_values = data[x_col]
        y_values = data[y_col]
        ax.grid(axis="both", linestyle="dotted", linewidth=1)

        mask = masks["default"]
        ax.scatter(x_values[mask], y_values[mask], **params["default"])

        mask = masks["highlight"]
        _annotated_scatter(
            x_values=data[x_col][mask],
            y_values=data[y_col][mask],
            labels=data[annotation_column][mask],
            ax=ax,
            scatter_kws=params["highlight"],
        )

        ax.set_xlabel(x_variable)
        ax.set_ylabel(y_variable)

    fig.tight_layout()
    return fig, axes


def expression_comparison(
    qtable: Qtable,
    experiment_pair: list[str],
    comparison_tag: str = " vs ",
    plot_average_expression: bool = False,
    special_proteins: Optional[list[str]] = None,
    exclude_invalid: bool = True,
) -> (plt.Figure, list[plt.Axes]):
    """Generates an expression comparison plot for two experiments.

    The subplot in the middle displays the average expression of the two experiments on
    the y-axis and the log fold change on the x-axis. The subplots on the left and right
    display entries with only missing values in one of the two experiments.

    Args:
        qtable: A Qtable instance, which data is used for the PCA analysis.
        experiment_pair: The names of the two experiments that will be compared,
            experiments must be present in qtable.design.
        comparison_tag: String used in comparison columns to separate a pair of
            experiments; default " vs ", which corresponds to the MsReport convention.
        plot_average_expression: Optional, if True plot average expression instead of
            maxium expression. Default False.
        special_proteins: Optional, allows to specify a list of entries from the
            "Representative Protein" column to be annotated. Entries are annotated with
            values from the "Gene Name" column if present, otherwise from the
            "Representative Protein" column.
        exclude_invalid: Optional, if True rows are filtered according to the Boolean
            entries of "Valid"; default True.

    Returns:
        A matplotlib Figure objects and a list of three Axes objects containing the
        expression comparison plots.
    """
    exp_1, exp_2 = experiment_pair
    comparison_group = comparison_tag.join(experiment_pair)
    special_proteins = special_proteins if special_proteins is not None else []
    qtable_data = qtable.get_data(exclude_invalid=exclude_invalid)
    annotation_column = (
        "Gene name" if "Gene name" in qtable_data.columns else "Representative protein"
    )
    total_scatter_area = 5000
    params = {
        "highlight": {
            "s": 10,
            "color": "#FAB74E",
            "edgecolor": "#000000",
            "lw": 0.2,
            "zorder": 3,
        },
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

    width_ratios = [1, 5, 1]
    fig, axes = plt.subplots(
        1, 3, figsize=[6, 4], sharey=True, gridspec_kw={"width_ratios": width_ratios}
    )

    # Plot values quantified in both experiments
    ax = axes[1]
    values = qtable_data[mask_both]
    s = scattersize(values, total_scatter_area)
    x_variable = f"Ratio [log2]"
    y_variable = (
        f"Average expression" if plot_average_expression else f"Maximum expression"
    )
    x_col = " ".join([x_variable, comparison_group])
    y_col = " ".join([y_variable, comparison_group])
    x_values = values[x_col]
    y_values = values[y_col]
    ax.grid(axis="both", linestyle="dotted", linewidth=1)
    ax.scatter(x_values, y_values, s=s, alpha=0.75, color="#A0A0A0", zorder=2)
    highlight_mask = values["Representative protein"].isin(special_proteins)
    _annotated_scatter(
        x_values=x_values[highlight_mask],
        y_values=y_values[highlight_mask],
        labels=values[annotation_column][highlight_mask],
        ax=ax,
        scatter_kws=params["highlight"],
    )

    ax.set_xlabel(x_variable, fontsize=9)
    ax.set_title(comparison_group, fontsize=12)
    ax.set_ylabel(y_variable)

    # Plot values quantified only in one experiment
    for ax, mask, exp in [(axes[2], only_exp_1, exp_1), (axes[0], only_exp_2, exp_2)]:
        y_variable = f"Expression {exp}"
        values = qtable_data[mask]
        s = scattersize(values, total_scatter_area)
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=UserWarning)
            try:
                sns.swarmplot(
                    y=values[y_variable],
                    size=np.sqrt(s * 2),
                    marker="o",
                    alpha=0.75,
                    color="#A0A0A0",
                    edgecolor="none",
                    ax=ax,
                )
                fig.canvas.draw()
            except UserWarning:
                ax.cla()
                sns.stripplot(
                    y=values[y_variable],
                    jitter=True,
                    size=np.sqrt(s * 2),
                    marker="o",
                    alpha=0.75,
                    color="#A0A0A0",
                    edgecolor="none",
                    ax=ax,
                )
                ax.set_xlim(-0.2, 0.2)
            finally:
                xlim = ax.get_xlim()
                highlight_mask = values["Representative protein"].isin(special_proteins)
                offsets = ax.collections[0].get_offsets()[highlight_mask]
                _annotated_scatter(
                    x_values=offsets[:, 0],
                    y_values=offsets[:, 1],
                    labels=values[annotation_column][highlight_mask],
                    ax=ax,
                )
                ax.set_xlim(xlim)

        ax.grid(axis="y", linestyle="dotted", linewidth=1)
        ax.set_ylabel(y_variable)
    axes[0].set_title(f"Absent in\n{exp_1}", fontsize=9)
    axes[2].set_title(f"Absent in\n{exp_2}", fontsize=9)

    fig.tight_layout()
    return fig, axes


def box_and_bars(
    box_values: Iterable[Iterable[float]],
    bar_values: Iterable[float],
    group_names: list[str],
    colors: Optional[list[str]] = None,
) -> (plt.Figure, list[plt.Axes]):
    """Generates a figure with horizontally aligned box and bar subplots.

    In the top subplot the 'box_values' are displayed as box plots, in lower subplot the
    'bar_values' are displayed as bar plots. The figure width is automatically adjusted
    to the number of groups that will be plotted. The length of group_names must be the
    same as the length of the of the 'bar_values' and the number of iterables from
    'box_values'. Each group from 'box_values' and 'bar_values' is horizontally aligned
    between the two subplots.

    Args:
        box_values: A sequence of sequences that each contain y values for generating a
            box plot.
        bar_values: A sequence of y values for generating bar plots.
        group_names: Used to label groups from box and bar plots.
        colors: Sequence of hex color codes for each group that is used for the boxes of
            the box and bar plots. Must be the same length as group names. If 'colors'
            is None, boxes are colored in light grey.

    Returns:
        A matplotlib Figure and a list of Axes objects containing the box and bar plots.
    """
    assert len(box_values) == len(bar_values) == len(group_names)
    assert colors is None or len(colors) == len(group_names)
    if colors is None:
        colors = ["#D0D0D0" for _ in group_names]

    num_samples = len(group_names)
    x_values = range(num_samples)
    width = 0.8
    xlim = (-1 + 0.15, num_samples - 0.15)
    figwidth = (num_samples * 0.25) + 1.2
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


def expression_clustermap(
    qtable: Qtable,
    exclude_invalid: bool = True,
) -> sns.matrix.ClusterGrid:
    """Plot sample expression values as a hierarchically-clustered heatmap.

    Args:
        qtable: A Qtable instance, which expression data is used for the heatmap.
        exclude_invalid: Optional, if True rows are filtered according to the Boolean
            entries of "Valid"; default True.

    Returns:
        A seaborn ClusterGrid instance, which has a `savefig` method that can be used
        for saving the figure.
    """
    samples = qtable.get_samples()
    experiments = qtable.get_experiments()

    data = qtable.make_expression_table(samples_as_columns=True)
    data = data[samples]
    for sample in samples:
        data.loc[qtable.data[f"Missing {sample}"], sample] = 0
    imputed_values = qtable.data[[f"Missing {sample}" for sample in samples]].to_numpy()

    if exclude_invalid:
        data = data[qtable.data["Valid"]]
        imputed_values = imputed_values[qtable.data["Valid"]]

    color_wheel = ColorWheelDict()
    _ = [color_wheel[exp] for exp in experiments]
    sample_colors = [color_wheel[qtable.get_experiment(sample)] for sample in samples]
    figsize = (0.3 + len(samples) * 0.4, 5)

    # Generate the plot
    cluster_grid = sns.clustermap(
        data,
        col_colors=sample_colors,
        cmap="magma",
        yticklabels=False,
        mask=imputed_values,
        figsize=figsize,
        metric="euclidean",
        method="median",
    )
    cluster_grid.ax_row_dendrogram.set_visible(False)

    # Add background color and spines
    cluster_grid.ax_heatmap.set_facecolor("#F9F9F9")
    for _, spine in cluster_grid.ax_heatmap.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
    return cluster_grid


def pvalue_histogram(
    qtable: Qtable,
    pvalue_tag: str = "P-value",
    comparison_tag: str = " vs ",
    experiment_pairs: Optional[Iterable[Iterable[str]]] = None,
    exclude_invalid: bool = True,
) -> (plt.Figure, list[plt.Axes]):
    """Generates p-value histograms for one or multiple experiment comparisons.

    Histograms are generated with 20 bins of size 0.05. The p-values of each experiment
    comparison are plotted on a separate subplot.

    Args:
        qtable: A Qtable instance, which data is used for the p-value histograms.
        pvalue_tag: String used for matching the pvalue columns; default "P-value",
            which corresponds to the MsReport convention.
        comparison_tag: String used in comparison columns to separate a pair of
            experiments; default " vs ", which corresponds to the MsReport convention.
        experiment_pairs: Optional, list of experiment pairs that will be used for
            plotting. For each experiment pair a p-value column must exists that follows
            the format f"{pvalue_tag} {experiment_1}{comparison_tag}{experiment_2}".
            If None, all experiment comparisons that are found in qtable.data are used.
        exclude_invalid: Optional, if True rows are filtered according to the Boolean
            entries of "Valid"; default True.

    Returns:
        A matplotlib Figure and a list of Axes objects, containing the p-value plots.
    """
    data = qtable.get_data(exclude_invalid=exclude_invalid)

    # Find all experiment pairs
    if experiment_pairs is None:
        experiment_pairs = []
        for experiment_pair in itertools.permutations(qtable.get_experiments(), 2):
            comparison_group = comparison_tag.join(experiment_pair)
            comparison_column = f"{pvalue_tag} {comparison_group}"
            if comparison_column in data.columns:
                experiment_pairs.append(experiment_pair)

    num_plots = len(experiment_pairs)

    figwidth = (num_plots * 1.8) + -0.6
    figheight = 2.5
    figsize = (figwidth, figheight)

    fig, axes = plt.subplots(1, num_plots, figsize=figsize, sharex=True, sharey=True)
    axes = axes if isinstance(axes, Iterable) else (axes,)
    fig.subplots_adjust(wspace=0.5)

    bins = np.arange(0, 1.01, 0.05)
    for plot_number, experiment_pair in enumerate(experiment_pairs):
        ax = axes[plot_number]
        comparison_group = comparison_tag.join(experiment_pair)
        comparison_column = f"{pvalue_tag} {comparison_group}"
        p_values = data[comparison_column]
        sns.histplot(
            p_values,
            bins=bins,
            ax=ax,
            element="bars",
            zorder=2,
            color="#FAB74E",
            edgecolor="#FFFFFF",
            linewidth=0.7,
        )

        # Adjust x- and y-axis
        ax.set_xlabel(None)
        ax.set_xticks(np.arange(0, 1.01, 0.5))
        ax.tick_params(labelsize=9)
        if plot_number > 0:
            ax.tick_params(axis="y", color="none")

        # Add second label
        ax2 = ax.twinx()
        ax2.set_yticks([])
        ax2.set_ylabel(comparison_group, fontsize=9)

        # Adjust spines
        sns.despine(top=True, right=True)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#000000")
            ax.spines[spine].set_linewidth(1)

        # Adjust grid
        ax.grid(False, axis="x")
        ax.grid(axis="y", linestyle="dashed", linewidth=1, color="#cccccc", zorder=1)

    axes[0].set_ylabel(f"{pvalue_tag} count")
    ax.set_xlim(-0.05, 1.05)

    return fig, axes


def _annotated_scatter(x_values, y_values, labels, ax=None, scatter_kws=None) -> None:
    ax = plt.gca() if ax is None else ax
    text_params = {
        "force_text": 0.15,
        "arrowprops": dict(arrowstyle="-", color="#ebae34", lw=0.75),
        "lim": 100,
    }
    if scatter_kws is None:
        scatter_kws = {
            "s": 10,
            "color": "#FAB74E",
            "edgecolor": "#000000",
            "lw": 0.2,
            "zorder": 3,
        }

    texts = []
    for x, y, text in zip(x_values, y_values, labels):
        texts.append(ax.text(x, y, text, fontdict={"fontsize": 9}))

    adjustText.adjust_text(texts, ax=ax, **text_params)
    ax.scatter(x_values, y_values, **scatter_kws)
