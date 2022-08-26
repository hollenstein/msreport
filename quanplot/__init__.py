import itertools

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import statsmodels.stats.multitest


def missing_values_per_category(
    method_tables: dict[str, pd.DataFrame],
    experiments: str,
    categories: list[str] = None,
    method_colors: dict[str, str] = None,
):
    """Plot analysis of missing values per experiment"""
    if not categories:
        categories = set()
        for table in method_tables.values():
            categories.update(table["Category"])
    categories = sorted(categories)

    figures = []
    for category in categories:
        tables = {}
        for method, expr in method_tables.items():
            tables[method] = expr[expr["Category"] == category]
        fig, axes = missing_values(tables, experiments, method_colors=method_colors)
        ylabel = axes[0].yaxis.get_label().get_text()
        axes[0].set_ylabel(f"{category}: {ylabel}")
        figures.append([fig, axes])
    return figures


def missing_values(
    method_tables: dict[str, pd.DataFrame],
    experiments,
    method_colors: dict[str, str] = None,
):

    if not method_colors:
        color_cycler = itertools.cycle(sns.color_palette())
        method_colors = dict(zip(method_tables, color_cycler))

    num_methods = len(method_tables)
    num_experiments = len(experiments)

    barwidth = 0.8 / num_methods
    legendwidth = 1.5
    # figwidth = max(num_experiments * num_methods * 0.8 + 0.5 + legendwidth, 5)
    figwidth = (num_experiments * 0.8) + (num_experiments * num_methods * 0.6)
    figwidth = max(figwidth, 1.5) + legendwidth
    figsize = (figwidth, 3.5)
    xtick_labels = ["No missing", "Some missing", "All missing"]

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, num_experiments, figsize=figsize, sharey=True)
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


def cv_distribution_per_category(
    method_tables: dict[str, pd.DataFrame],
    exp_to_samples: dict[str, str],
    categories: list[str] = None,
    normed: bool = False,
    max_missing: int = 10,
    method_colors: dict[str, str] = None,
):
    """Plot analysis of missing values per experiment"""
    if not categories:
        categories = set()
        for table in method_tables.values():
            categories.update(table["Category"])
    categories = sorted(categories)

    figures = []
    for category in categories:
        tables = {}
        for method, expr in method_tables.items():
            tables[method] = expr[expr["Category"] == category]
        fig, axes = cv_distribution(
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


def cv_distribution(
    method_tables: dict[str, pd.DataFrame],
    exp_to_samples: dict[str, str],
    normed: bool = False,
    max_missing: int = 10,
    method_colors: dict[str, str] = None,
):

    if not method_colors:
        color_cycler = itertools.cycle(sns.color_palette())
        method_colors = dict(zip(method_tables, color_cycler))

    num_experiments = len(exp_to_samples)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(
        1, num_experiments, figsize=[num_experiments * 3, 3], sharex=True
    )
    for exp_num, exp in enumerate(exp_to_samples):
        ax = axes[exp_num]
        for method_num, (method, table) in enumerate(method_tables.items()):
            data = table[table["Valid"]]
            column = " ".join(["Missing", exp])
            data = data[data[column] <= max_missing]
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


def ratio_dist_per_category(
    method_tables: dict[str, pd.DataFrame],
    comparison_group: list[str],
    expected_ratios: dict[str, float] = None,
    max_missing: int = 10,
    method_colors: dict[str, str] = None,
):

    if not method_colors:
        color_cycler = itertools.cycle(sns.color_palette())
        method_colors = dict(zip(method_tables, color_cycler))

    categories = set()
    for table in method_tables.values():
        categories.update(table["Category"])
    categories = sorted(categories)

    num_methods = len(method_tables)
    exp1, exp2 = comparison_group

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, figsize=[8, 5], sharex=True)
    fig.suptitle(f"{exp1} vs. {exp2}")
    for method_num, (method, table) in enumerate(method_tables.items()):
        for cat_num, category in enumerate(categories):
            ax = axes[cat_num]
            data = table[table["Valid"] & (table["Category"] == category)]
            for exp in [exp1, exp2]:
                column = " ".join(["Missing", exp])
                data = data[data[column] <= max_missing]
            ratios = data[exp1] - data[exp2]
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


def precision_tp(
    method_tables: dict[str, pd.DataFrame],
    comparison_group: list[str],
    exp_to_samples: dict[str, str],
    bg_category: str,
    fg_category: str,
    max_missing: int = 10,
    method_colors: dict[str, str] = None,
):

    if not method_colors:
        color_cycler = itertools.cycle(sns.color_palette())
        method_colors = dict(zip(method_tables, color_cycler))

    exp1, exp2 = comparison_group
    samples1 = exp_to_samples[exp1]
    samples2 = exp_to_samples[exp2]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(1, figsize=[6, 6], sharex=True)
    fig.suptitle(f"{exp1} vs. {exp2}")
    for method_num, (method, table) in enumerate(method_tables.items()):
        data = table[table["Valid"]].copy()
        for exp in [exp1, exp2]:
            column = " ".join(["Missing", exp])
            data = data[data[column] <= max_missing]

        stats, pvalues = scipy.stats.ttest_ind(data[samples1], data[samples2], axis=1)
        _, pvalues_corr, _, _ = statsmodels.stats.multitest.multipletests(
            pvalues, method="fdr_bh", is_sorted=False
        )
        data["pvalues"] = np.log10(pvalues_corr)
        data.sort_values(by="pvalues", inplace=True)

        cat_counter = {bg_category: 0, fg_category: 0}
        curve_values = {"pvalue": [], "precision": [], "TP": []}
        for pvalue, cat in zip(data["pvalues"], data["Category"]):
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
        for pvalue, precision, tp in zip(
            curve_values["pvalue"], curve_values["precision"], curve_values["TP"]
        ):
            if pvalue >= -2:
                if method_num == 0:
                    ax.scatter(
                        tp,
                        precision,
                        marker="D",
                        s=35,
                        zorder=3,
                        lw=1.8,
                        edgecolor=method_colors[method],
                        color="#f0f0f0",
                        label="FDR corrected p-value at 0.01",
                    )
                else:
                    ax.scatter(
                        tp,
                        precision,
                        marker="D",
                        s=35,
                        zorder=3,
                        lw=1.8,
                        edgecolor=method_colors[method],
                        color="#f0f0f0",
                    )
                break
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


def cumulative_pvalues(
    method_tables: dict[str, pd.DataFrame],
    comparison_group: list[str],
    exp_to_samples: dict[str, str],
    bg_category: str,
    fg_category: str,
    max_missing: int = 10,
    method_colors: dict[str, str] = None,
):

    if not method_colors:
        color_cycler = itertools.cycle(sns.color_palette())
        method_colors = dict(zip(method_tables, color_cycler))

    exp1, exp2 = comparison_group
    samples1 = exp_to_samples[exp1]
    samples2 = exp_to_samples[exp2]

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=[8, 4], sharex=True)
    fig.suptitle(f"{exp1} vs. {exp2}")
    for method, table in method_tables.items():
        data = table[table["Valid"]].copy()
        for exp in [exp1, exp2]:
            column = " ".join(["Missing", exp])
            data = data[data[column] <= max_missing]

        stats, pvalues = scipy.stats.ttest_ind(data[samples1], data[samples2], axis=1)
        _, pvalues_corr, _, _ = statsmodels.stats.multitest.multipletests(
            pvalues, method="fdr_bh", is_sorted=False
        )
        data["pvalues"] = np.log10(pvalues_corr)
        for cat_num, category in enumerate([fg_category, bg_category]):
            ax = axes[cat_num]
            pvalues = data["pvalues"][data["Category"] == category]
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
