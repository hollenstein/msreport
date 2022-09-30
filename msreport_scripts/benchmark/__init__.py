import itertools
from typing import Optional, Iterable

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import msreport
import msreport.plot


def missing_values_per_category(
    qtables: dict[str, msreport.Qtable],
    experiments: Optional[list] = None,
    categories: Optional[list] = None,
    method_colors: dict[str, str] = None,
) -> list[(plt.Figure, list[plt.Axes])]:
    """Bar plot to analyze the completeness of quantification between different methods.

    Requires several columns to be present in the Qtable instances. Qtable entries are
    split into groups according to values in the "Category" column, each group is
    plotted separately. The columns "Missing experiment_name", which is added by calling
    msreport.analyze.analyze_missingness(qtable). The column "Valid" is used for
    removing non-valid entries, such as contaminants.

    Args:
        qtables: Dictionary that contains the Qtable of the different methods.
        experiments: Optional, specifies which experiments should be plotted.
        categories: Optional, specifies which categoires should be plotted.
        method_colors: Optional, allows specifying a hex color for each method.
    Returns:
        A list of tuples containing a matplotlib Figure object and a list of Axes
        objects. Each category generates one figure entry and each experiment generates
        one Axes object.
    """
    methods = list(qtables.keys())
    if experiments is None:
        experiments = qtables[methods[0]].get_experiments()
    if categories is None:
        categories = qtables[methods[0]].data["Category"].unique().tolist()
    if method_colors is None:
        method_colors = msreport.plot.ColorWheelDict()

    figures = []
    for category in categories:
        tables = {}
        for method in methods:
            data = qtables[method].data
            tables[method] = data[data["Category"] == category]
        fig, axes = _missing_values(tables, experiments, method_colors=method_colors)
        ylabel = axes[0].yaxis.get_label().get_text()
        axes[0].set_ylabel(f"{category}: {ylabel}")
        figures.append([fig, axes])
    return figures


def cv_distribution_per_category(
    qtables: dict[str, msreport.Qtable],
    experiments: Optional[list] = None,
    categories: Optional[list] = None,
    method_colors: dict[str, str] = None,
    normed: bool = False,
    max_missing: int = 99,
) -> list[(plt.Figure, list[plt.Axes])]:
    """Cumulative density plot of the coefficient of variation of different methods.

    Requires several columns to be present in the Qtable instances. Qtable entries are
    split into groups according to values in the "Category" column, each group is
    plotted separately. The columns "Missing experiment_name", which is added by calling
    msreport.analyze.analyze_missingness(qtable). The column "Valid" is used for
    removing non-valid entries, such as contaminants.

    Args:
        qtables: Dictionary that contains the Qtable of the different methods.
        experiments: Optional, specifies which experiments should be plotted.
        categories: Optional, specifies which categoires should be plotted.
        method_colors: Optional, allows specifying a hex color for each method.
        density: If true bins are normalized to the total number of counts and the bin
            width, so that the area under the histogram integrates to 1
        max_missing: Optional, removes entries with more missing values per experiment
            than 'max_missing'.

    Returns:
        A list of tuples containing a matplotlib Figure object and a list of Axes
        objects. Each category generates one figure entry and each experiment generates
        one Axes object.
    """
    methods = list(qtables.keys())
    if experiments is None:
        experiments = qtables[methods[0]].get_experiments()
    if categories is None:
        categories = qtables[methods[0]].data["Category"].unique().tolist()
    if method_colors is None:
        method_colors = msreport.plot.ColorWheelDict()

    exp_to_samples = {exp: qtables[methods[0]].get_samples(exp) for exp in experiments}
    figures = []
    for category in categories:
        tables = {}
        for method in methods:
            features = ["Category", "Valid"]
            features.extend([f"Missing {exp}" for exp in experiments])
            table = qtables[method].make_expression_table(
                samples_as_columns=True, features=features
            )
            tables[method] = table[table["Valid"] & (table["Category"] == category)]
        fig, axes = _cv_distribution(
            tables,
            exp_to_samples,
            normed=normed,
            max_missing=max_missing,
            method_colors=method_colors,
        )
        ylabel = axes[0].yaxis.get_label().get_text()
        axes[0].set_ylabel(f"{category}: {ylabel}")
        fig.tight_layout()
        figures.append([fig, axes])
    return figures


def ratio_dist_per_category(
    qtables: dict[str, msreport.Qtable],
    comparison_group: list[str, str],
    expected_ratios: dict[str, float] = None,
    categories: Optional[list] = None,
    method_colors: dict[str, str] = None,
    max_missing: int = 99,
) -> (plt.Figure, list[plt.Axes]):
    """Density plot comparing log2 ratios of two experiments between different methods.

    Requires that average experiment expression values are calculated. Which can be done
    by calling msreport.analyze.calculate_experiment_means(qtable).

    Requires several columns to be present in the Qtable instances. Qtable entries are
    split into groups according to values in the "Category" column, each group is
    plotted separately. The columns "Missing experiment_name", which is added by calling
    msreport.analyze.analyze_missingness(qtable). The column "Valid" is used for
    removing non-valid entries, such as contaminants.

    Args:
        qtables: Dictionary that contains the Qtable of the different methods.
        comparison_group: Specifies a pair of experiments that will be used to calculate
            log2 ratios.
        categories: Optional, specifies which categoires should be plotted.
        method_colors: Optional, allows specifying a hex color for each method.
        max_missing: Optional, removes entries with more missing values per experiment
            than 'max_missing'.

    Returns:
        A matplotlib Figure object and a list of Axes objects. Each category generates
        one Axes object.
    """
    methods = list(qtables.keys())
    if categories is None:
        categories = qtables[methods[0]].data["Category"].unique().tolist()
    if method_colors is None:
        method_colors = msreport.plot.ColorWheelDict()

    num_categories = len(categories)
    num_methods = len(methods)
    exp1, exp2 = comparison_group

    sns.set_style("whitegrid")
    figheight = num_categories * 2.25 + 0.5
    fig, axes = plt.subplots(num_categories, figsize=[8, figheight], sharex=True)
    axes = axes if isinstance(axes, Iterable) else (axes,)

    fig.suptitle(f"{exp1} vs. {exp2}")
    for method_num, method in enumerate(methods):
        qtable = qtables[method]
        table = qtable.data[qtable.data["Valid"]]
        for exp in [exp1, exp2]:
            column = " ".join(["Missing", exp])
            table = table[table[column] <= max_missing]

        for cat_num, category in enumerate(categories):
            ax = axes[cat_num]
            data = table[(table["Category"] == category)]
            ratios = data[f"Expression {exp1}"] - data[f"Expression {exp2}"]
            sns.kdeplot(
                ratios,
                ax=ax,
                common_norm=True,
                label=method,
                color=method_colors[method],
            )
            if expected_ratios and (method_num == num_methods - 1):
                line_y = ax.get_ylim()
                line_x = (expected_ratios[category], expected_ratios[category])
                xlim = (
                    min(expected_ratios.values()) - 1.5,
                    max(expected_ratios.values()) + 1.5,
                )
                ax.plot(line_x, line_y, color="#333333", zorder=1, lw=1.2)
                ax.set_xlim(xlim)
                ax.set_ylim(line_y)
            ax.set_yticks([])
            ax.set_ylabel(f"{category}: Protein density")
            ax.spines["bottom"].set_color("#000000")
            ax.spines["bottom"].set_linewidth(1.2)
            ax.grid(axis="x", linestyle="dashed", linewidth=1)
    axes[-1].set_xlabel("Ratios [log2]")
    axes[0].legend(loc="upper right")
    sns.despine(top=True, right=True, left=True)
    return fig, axes


def cumulative_pvalues(
    qtables: dict[str, msreport.Qtable],
    comparison_group: list[str],
    categories: Optional[list] = None,
    method_colors: dict[str, str] = None,
    max_missing: int = 99,
) -> (plt.Figure, list[plt.Axes]):
    """Compares p-values of different categories between different methods.

    Requires that Adjusted p-values have been calculated for the 'comparison_group',
    which can be done by calling msreport.analyze.calculate_two_group_limma().

    Requires several columns to be present in the Qtable instances. Qtable entries are
    split into groups according to values in the "Category" column, each group is
    plotted separately. The columns "Missing experiment_name", which is added by calling
    msreport.analyze.analyze_missingness(qtable). The column "Valid" is used for
    removing non-valid entries, such as contaminants.

    Args:
        qtables: Dictionary that contains the Qtable of the different methods.
        comparison_group: Specifies the experiment pair that will be compared.
        categories: Optional, specifies which categoires should be plotted.
        method_colors: Optional, allows specifying a hex color for each method.
        max_missing: Optional, removes entries with more missing values per experiment
            than 'max_missing'.

    Returns:
        A matplotlib Figure object and a list of Axes objects. Each category generates
        one Axes object.
    """
    methods = list(qtables.keys())
    if categories is None:
        categories = qtables[methods[0]].data["Category"].unique().tolist()
    if method_colors is None:
        method_colors = msreport.plot.ColorWheelDict()

    num_categories = len(categories)
    exp1, exp2 = comparison_group
    comparison_tag = " vs ".join(comparison_group)
    pvalue_column = f"Adjusted p-value {comparison_tag}"

    sns.set_style("whitegrid")
    figwidth = num_categories * 3.75 + 0.5
    fig, axes = plt.subplots(1, num_categories, figsize=[figwidth, 4], sharex=True)
    fig.suptitle(f"{exp1} vs. {exp2}")
    axes = axes if isinstance(axes, Iterable) else (axes,)

    for method in methods:
        qtable = qtables[method]
        table = qtable.data[qtable.data["Valid"]]
        for exp in [exp1, exp2]:
            column = " ".join(["Missing", exp])
            table = table[table[column] <= max_missing]

        for cat_num, category in enumerate(categories):
            ax = axes[cat_num]
            data = table[(table["Category"] == category)]
            pvalues = np.log10(data[pvalue_column])
            bins = np.linspace(-10, 0.2, 200)
            ax.hist(
                pvalues,
                bins=bins,
                cumulative=True,
                histtype="step",
                density=False,
                label=method,
                color=method_colors[method],
            )
            for spine in ["bottom", "left"]:
                ax.spines[spine].set_color("#000000")
                ax.spines[spine].set_linewidth(1)
            ax.grid(axis="both", linestyle="dotted", linewidth=1.2)
            ax.set_xlabel("Corrected p-values [log10]")
            ax.set_ylabel(f"{category}: # proteins")
            ax.set_xlim(-5, 0.1)
    axes[-1].legend(loc="upper left")
    fig.tight_layout()
    return fig, axes


def precision_tp(
    qtables: dict[str, msreport.Qtable],
    comparison_group: list[str],
    bg_category: str,
    fg_category: str,
    method_colors: dict[str, str] = None,
    max_missing: int = 99,
) -> (plt.Figure, plt.Axes):
    """Generates a precision/true-positive curve for each method.

    Requires an experiment with known truth about which values should be differentially
    expressed (foreground) and which should be constant (background); this must be
    encoded into two categories

    Requires that Adjusted p-values have been calculated for the 'comparison_group',
    which can be done by calling msreport.analyze.calculate_two_group_limma().

    Requires several columns to be present in the Qtable instances. Qtable entries are
    split into groups according to values in the "Category" column. The columns
    "Missing experiment_name", which is added by calling
    msreport.analyze.analyze_missingness(qtable). The column "Valid" is used for
    removing non-valid entries, such as contaminants.

    Args:
        qtables: Dictionary that contains the Qtable of the different methods.
        comparison_group: Specifies the experiment pair that will be compared.
        bg_category: "Category" value of entries from the background category.
        fg_category: "Category" value of entries from the foreground category.
        method_colors: Optional, allows specifying a hex color for each method.
        max_missing: Optional, removes entries with more missing values per experiment
            than 'max_missing'.

    Returns:
        A matplotlib Figure object and an Axes objects.
    """
    methods = list(qtables.keys())
    if method_colors is None:
        method_colors = msreport.plot.ColorWheelDict()

    exp1, exp2 = comparison_group
    comparison_tag = " vs ".join(comparison_group)
    pvalue_column = f"Adjusted p-value {comparison_tag}"

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, figsize=[6, 6], sharex=True)
    fig.suptitle(f"{exp1} vs. {exp2}")

    for method_num, method in enumerate(methods):
        qtable = qtables[method]
        table = qtable.data[qtable.data["Valid"]]
        for exp in [exp1, exp2]:
            column = " ".join(["Missing", exp])
            table = table[table[column] <= max_missing]
        data = table.copy()
        data[pvalue_column] = np.log10(data[pvalue_column])
        data.sort_values(by=pvalue_column, inplace=True)

        cat_counter = {bg_category: 0, fg_category: 0}
        curve_values = {"pvalue": [], "precision": [], "TP": []}
        for pvalue, cat in zip(data[pvalue_column], data["Category"]):
            cat_counter[cat] += 1
            TP = cat_counter[fg_category]
            FP = cat_counter[bg_category]
            precision = TP / (TP + FP) * 100
            curve_values["precision"].append(precision)
            curve_values["TP"].append(TP)
            curve_values["pvalue"].append(pvalue)

        ax.plot(
            curve_values["TP"],
            curve_values["precision"],
            color=method_colors[method],
            label=method,
        )

        index_cutoff = np.argmax(np.array(curve_values["pvalue"]) >= -2)
        label = "FDR corrected p-value at 0.01" if method_num == 0 else None
        ax.scatter(
            curve_values["TP"][index_cutoff],
            curve_values["precision"][index_cutoff],
            marker="D",
            s=35,
            zorder=3,
            lw=1.8,
            edgecolor=method_colors[method],
            color="#f0f0f0",
            label=label,
        )

    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#000000")
        ax.spines[spine].set_linewidth(1)
    ax.grid(axis="both", linestyle="dotted", linewidth=1.2)
    ax.set_xlabel("# True positive")
    ax.set_ylabel("Precision [%]")
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(75, 100.5)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig, ax


def _missing_values(
    method_tables: dict[str, pd.DataFrame],
    experiments: list[str],
    method_colors: dict[str, str],
) -> (plt.Figure, list[plt.Axes]):
    num_methods = len(method_tables)
    num_experiments = len(experiments)

    barwidth = 0.8 / num_methods
    legendwidth = 1.5
    figwidth = (num_experiments * 0.8) + (num_experiments * num_methods * 0.6)
    figwidth = max(figwidth, 1.5) + legendwidth
    figsize = (figwidth, 3.5)
    xtick_labels = ["No missing", "Some missing", "All missing"]

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, num_experiments, figsize=figsize, sharey=True)
    axes = axes if isinstance(axes, Iterable) else (axes,)

    for exp_num, exp in enumerate(experiments):
        for method_num, method in enumerate(method_tables):
            ax = axes[exp_num]
            table = method_tables[method]
            data = table
            if "Valid" in data.columns:
                data = data[data["Valid"]]

            exp_missing = data[f"Missing {exp}"]
            missing_none = (exp_missing == 0).sum()
            missing_some = ((exp_missing > 0) & (exp_missing < 3)).sum()
            missing_all = (exp_missing == 3).sum()

            y = [missing_none, missing_some, missing_all]
            x = [i + method_num * barwidth for i in np.arange(3)]
            ax.bar(x, y, width=barwidth, label=method, color=method_colors[method])
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

    # Add a legend below the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1, 0.91), ncol=1)
    figure_space_for_legend = 1 - (1.5 / figwidth)
    fig.tight_layout(rect=[0, 0, figure_space_for_legend, 1])

    return fig, axes


def _cv_distribution(
    method_tables: dict[str, pd.DataFrame],
    exp_to_samples: dict[str, str],
    normed: bool = False,
    max_missing: int = 99,
    method_colors: dict[str, str] = None,
) -> (plt.Figure, list[plt.Axes]):
    if not method_colors:
        color_cycler = itertools.cycle(sns.color_palette())
        method_colors = dict(zip(method_tables, color_cycler))

    num_experiments = len(exp_to_samples)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(
        1, num_experiments, figsize=[num_experiments * 3, 3], sharex=True
    )
    axes = axes if isinstance(axes, Iterable) else (axes,)

    for exp_num, exp in enumerate(exp_to_samples):
        ax = axes[exp_num]
        for method_num, (method, table) in enumerate(method_tables.items()):
            data = table[table[f"Missing {exp}"] <= max_missing]
            samples = exp_to_samples[exp]
            intensities = data[samples]
            std = np.std(intensities, axis=1)
            mean = np.mean(intensities, axis=1)
            cv = std / mean * 100
            bins = np.linspace(0, 2, 400)
            ax.hist(
                cv,
                bins=bins,
                cumulative=True,
                histtype="step",
                density=normed,
                lw=1.2,
                color=method_colors[method],
                label=method,
            )
            ax.set_title(exp)
            ax.legend(loc="lower right")
            ax.set_xlabel("Cv of protein intensities [log2]")
            ax.set_xlim(0, 1.5)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#000000")
            ax.spines[spine].set_linewidth(1)
        ax.grid(axis="both", linestyle="dotted", linewidth=1.2)

    sns.despine(top=True, right=True)
    axes[0].set_ylabel("# Proteins")
    return fig, axes
