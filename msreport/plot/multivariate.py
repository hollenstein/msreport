from typing import Any

import adjustText
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.decomposition
import sklearn.preprocessing

import msreport.helper
from msreport.qtable import Qtable

from .style import ColorWheelDict, with_active_style


@with_active_style
def sample_pca(
    qtable: Qtable,
    tag: str = "Expression",
    pc_x: str = "PC1",
    pc_y: str = "PC2",
    exclude_invalid: bool = True,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Figure to compare sample similarities with a principle component analysis.

    On the left subplots two PCA components of log2 transformed, mean centered intensity
    values are shown. On the right subplot the explained variance of the principle
    components is display as barplots.

    It is possible to use intensity columns that are either log-transformed or not. The
    intensity values undergo an automatic evaluation to determine if they are already
    in log-space, and if necessary, they are transformed accordingly.

    Args:
        qtable: A `Qtable` instance, which data is used for plotting.
        tag: A string that is used to extract intensity containing columns.
            Default "Expression".
        pc_x: Principle component to plot on x-axis of the scatter plot, default "PC1".
            The number of calculated principal components is equal to the number of
            samples.
        pc_y: Principle component to plot on y-axis of the scatter plot, default "PC2".
            The number of calculated principal components is equal to the number of
            samples.
        exclude_invalid: If True, rows are filtered according to the Boolean entries of
            the "Valid" column.

    Returns:
        A matplotlib Figure and a list of Axes objects, containing the PCA plots.
    """
    design = qtable.get_design()
    if design.shape[0] < 3:
        fig, ax = plt.subplots(1, 1, figsize=(2, 1.3))
        fig.suptitle(f'PCA of "{tag}" values', y=1.1)
        ax.text(
            0.5,
            0.5,
            "PCA analysis cannot\nbe performed with\nless than 3 samples",
            ha="center",
            va="center",
        )
        ax.grid(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        sns.despine(top=True, right=True, fig=fig)
        return fig, np.array([ax])

    table = qtable.make_sample_table(
        tag, samples_as_columns=True, exclude_invalid=exclude_invalid
    )
    table = table.replace({0: np.nan})
    table = table[np.isfinite(table).sum(axis=1) > 0]
    if not msreport.helper.intensities_in_logspace(table):
        table = np.log2(table)
    table[table.isna()] = 0

    table = table.transpose()
    sample_index = table.index.tolist()
    table = sklearn.preprocessing.scale(table, with_std=False)

    num_components = min(len(sample_index) - 1, 9)
    pca = sklearn.decomposition.PCA(n_components=num_components)
    components = pca.fit_transform(table)
    component_labels = ["PC{}".format(i + 1) for i in range(components.shape[1])]
    components_table = pd.DataFrame(
        data=components, columns=component_labels, index=sample_index
    )
    variance = pca.explained_variance_ratio_ * 100
    variance_lookup = dict(zip(component_labels, variance, strict=True))

    # Prepare colors
    color_wheel = ColorWheelDict()
    for exp in qtable.get_experiments():
        _ = color_wheel[exp]

    # Prepare figure
    num_legend_cols = 3  # math.ceil(len(qtable.get_experiments()) / 8)
    bar_width = 0.8
    bar_width_inches = 0.25
    x_padding = 0.25

    suptitle_space_inch = 0.4
    ax_height_inch = 2.7
    ax_width_inch = ax_height_inch
    ax_wspace_inch = 0.6
    bar_ax_width_inch = (num_components + (2 * x_padding)) * bar_width_inches
    width_ratios = [ax_width_inch, bar_ax_width_inch]

    fig_height = suptitle_space_inch + ax_height_inch
    fig_width = ax_height_inch + bar_ax_width_inch + ax_wspace_inch
    fig_size = (fig_width, fig_height)

    subplot_top = 1 - (suptitle_space_inch / fig_height)
    subplot_wspace = ax_wspace_inch / np.mean([ax_width_inch, bar_ax_width_inch])

    bar_half_width = 0.5
    lower_xbound = (0 - bar_half_width) - x_padding
    upper_xbound = (num_components - 1) + bar_half_width + x_padding

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
    fig.suptitle(f'PCA of "{tag}" values', y=1)

    # Comparison of two principle components
    ax = axes[0]
    texts = []
    for sample, data in components_table.iterrows():
        experiment = qtable.get_experiment(str(sample))
        label = design.loc[(design["Sample"] == sample), "Replicate"].tolist()[0]
        color = color_wheel[experiment]
        edge_color = color_wheel.modified_color(experiment, 0.4)
        ax.scatter(
            data[pc_x],
            data[pc_y],
            color=color,
            edgecolor=edge_color,
            lw=0.7,
            s=50,
            label=experiment,
        )
        texts.append(ax.text(data[pc_x], data[pc_y], label))
    adjustText.adjust_text(
        texts,
        force_text=0.15,
        expand_points=(1.4, 1.4),
        lim=20,
        ax=ax,
    )
    ax.set_xlabel(f"{pc_x} ({variance_lookup[pc_x]:.1f}%)")
    ax.set_ylabel(f"{pc_y} ({variance_lookup[pc_y]:.1f}%)")
    ax.grid(axis="both", linestyle="dotted")

    # Explained variance bar plot
    ax = axes[1]
    xpos = range(len(variance))
    ax.bar(xpos, variance, width=bar_width, color="#D0D0D0", edgecolor="#000000")
    ax.set_xticks(xpos)
    ax.set_xticklabels(
        component_labels,
        rotation="vertical",
        ha="center",
        size=plt.rcParams["axes.labelsize"],
    )
    ax.set_ylabel("Explained variance [%]")
    ax.grid(False, axis="x")
    ax.set_xlim(lower_xbound, upper_xbound)

    handles, labels = axes[0].get_legend_handles_labels()
    experiment_handles = dict(zip(labels, handles, strict=True))

    first_ax_bbox = axes[1].get_position()
    legend_xgap_inches = 0.25
    legend_ygap_inches = 0.03
    legend_bbox_x = first_ax_bbox.x1 + (legend_xgap_inches / fig.get_figwidth())
    legend_bbox_y = first_ax_bbox.y1 + (legend_ygap_inches / fig.get_figheight())
    handles, _ = axes[0].get_legend_handles_labels()
    num_legend_cols = np.ceil(len(qtable.get_experiments()) / 12)
    fig.legend(
        handles=experiment_handles.values(),
        loc="upper left",
        bbox_to_anchor=(legend_bbox_x, legend_bbox_y),
        title="Experiment",
        alignment="left",
        frameon=False,
        borderaxespad=0,
        ncol=num_legend_cols,
    )

    return fig, axes


@with_active_style
def expression_clustermap(
    qtable: Qtable,
    exclude_invalid: bool = True,
    remove_imputation: bool = True,
    mean_center: bool = False,
    cluster_samples: bool = True,
    cluster_method: str = "average",
) -> sns.matrix.ClusterGrid:
    """Plot sample expression values as a hierarchically-clustered heatmap.

    By default missing and imputed values are assigned an intensity value of 0 to
    perform the clustering. Once clustering is done, these values are removed from the
    heatmap, making them appear white.

    Args:
        qtable: A `Qtable` instance, which data is used for plotting.
        exclude_invalid: If True, rows are filtered according to the Boolean entries of
            the "Valid" column.
        remove_imputation: If True, imputed values are set to 0 before clustering.
            Defaults to True.
        mean_center: If True, the data is mean-centered before clustering. Defaults to
            False.
        cluster_samples: If True, sample order is determined by hierarchical clustering.
            Otherwise, the order is determined by the order of samples in the qtable
            design. Defaults to True.
        cluster_method: Linkage method to use for calculating clusters. See
            `scipy.cluster.hierarchy.linkage` documentation for more information.

    Raises:
        ValueError: If less than two samples are present in the qtable.

    Returns:
        A seaborn ClusterGrid instance. Note that ClusterGrid has a `savefig` method
        that can be used for saving the figure.
    """
    tag: str = "Expression"
    samples = qtable.get_samples()
    experiments = qtable.get_experiments()

    if len(samples) < 2:
        raise ValueError("At least two samples are required to generate a clustermap.")

    data = qtable.make_expression_table(samples_as_columns=True)
    data = data[samples]

    for sample in samples:
        if remove_imputation:
            data.loc[qtable.data[f"Missing {sample}"], sample] = 0
        data[sample] = data[sample].fillna(0)

    if not mean_center:
        # Hide missing values in the heatmap, making them appear white
        mask_values = qtable.data[
            [f"Missing {sample}" for sample in samples]
        ].to_numpy()
    else:
        mask_values = np.zeros(data.shape, dtype=bool)

    if exclude_invalid:
        data = data[qtable.data["Valid"]]
        mask_values = mask_values[qtable.data["Valid"]]

    color_wheel = ColorWheelDict()
    for exp in experiments:
        _ = color_wheel[exp]
    sample_colors = [color_wheel[qtable.get_experiment(sample)] for sample in samples]

    suptitle_space_inch = 0.4
    sample_width_inch = 0.27
    cbar_height_inch = 3
    cbar_width_inch = sample_width_inch
    cbar_gap_inch = sample_width_inch
    col_colors_height_inch = 0.12
    col_dendrogram_height_inch = 0.6 if cluster_samples else 0.0
    heatmap_height_inch = 3
    heatmap_width_inch = len(samples) * sample_width_inch

    fig_width = cbar_width_inch + heatmap_width_inch + cbar_gap_inch
    fig_height = (
        suptitle_space_inch
        + col_dendrogram_height_inch
        + col_colors_height_inch
        + heatmap_height_inch
    )
    fig_size = fig_width, fig_height

    heatmap_width = heatmap_width_inch / fig_width
    heatmap_x0 = 0
    heatmap_height = heatmap_height_inch / fig_height
    heatmap_y0 = 0
    col_colors_height = col_colors_height_inch / fig_height
    col_colors_y0 = heatmap_y0 + heatmap_height
    col_dendrogram_height = col_dendrogram_height_inch / fig_height
    col_dendrogram_y0 = col_colors_y0 + col_colors_height
    cbar_widh = cbar_width_inch / fig_width
    cbar_x0 = (heatmap_width_inch + cbar_gap_inch) / fig_width
    cbar_height = cbar_height_inch / fig_height
    cbar_y0 = col_colors_y0 - cbar_height

    heatmap_args: dict[str, Any] = {
        "cmap": "magma",
        "yticklabels": False,
        "figsize": fig_size,
    }
    if mean_center:
        data = data.sub(data.mean(axis=1), axis=0)
        heatmap_args.update({"vmin": -2.5, "vmax": 2.5, "center": 0, "cmap": "vlag"})

    # Generate the plot
    grid = sns.clustermap(
        data,
        col_cluster=cluster_samples,
        col_colors=sample_colors,
        row_colors=["#000000" for _ in range(len(data))],
        mask=mask_values,
        method=cluster_method,
        metric="euclidean",
        **heatmap_args,
    )
    # Reloacte clustermap axes to create a consistent layout
    grid.figure.suptitle(f'Hierarchically-clustered heatmap of "{tag}" values', y=1)
    grid.figure.delaxes(grid.ax_row_colors)
    grid.figure.delaxes(grid.ax_row_dendrogram)
    grid.ax_heatmap.set_position(
        [heatmap_x0, heatmap_y0, heatmap_width, heatmap_height]
    )
    grid.ax_col_colors.set_position(
        [heatmap_x0, col_colors_y0, heatmap_width, col_colors_height]
    )
    grid.ax_col_dendrogram.set_position(
        [heatmap_x0, col_dendrogram_y0, heatmap_width, col_dendrogram_height]
    )
    grid.ax_cbar.set_position([cbar_x0, cbar_y0, cbar_widh, cbar_height])

    # manually set xticks to guarantee that all samples are displayed
    if cluster_samples:
        sample_order = [samples[i] for i in grid.dendrogram_col.reordered_ind]
    else:
        sample_order = samples
    sample_ticks = np.arange(len(sample_order)) + 0.5
    grid.ax_heatmap.grid(False)
    grid.ax_heatmap.set_xticks(sample_ticks, labels=sample_order)
    grid.ax_heatmap.tick_params(
        axis="x", labelsize=plt.rcParams["axes.labelsize"], rotation=90
    )

    grid.ax_heatmap.set_facecolor("#F9F9F9")

    for ax in [grid.ax_heatmap, grid.ax_cbar, grid.ax_col_colors]:
        sns.despine(top=False, right=False, left=False, bottom=False, ax=ax)
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_linewidth(0.75)
    return grid
