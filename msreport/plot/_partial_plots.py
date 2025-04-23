from collections.abc import Iterable, Sequence
from typing import Optional

import adjustText
import matplotlib.pyplot as plt
import seaborn as sns

from .style import with_active_style


@with_active_style
def annotated_scatter(
    x_values,
    y_values,
    labels,
    ax=None,
    scatter_kws=None,
    text_kws=None,
) -> None:
    ax = plt.gca() if ax is None else ax
    if scatter_kws is None:
        scatter_kws = {}
    if text_kws is None:
        text_kws = {}
    text_params = {
        "force_text": 0.15,
        "arrowprops": {
            "arrowstyle": "-",
            "color": scatter_kws["color"],
            "lw": 0.75,
            "alpha": 0.5,
        },
        "lim": 100,
    }

    texts = []
    for x, y, text in zip(x_values, y_values, labels, strict=True):
        texts.append(ax.text(x, y, text, **text_kws))

    if texts:
        adjustText.adjust_text(texts, ax=ax, **text_params)  # type: ignore
        ax.scatter(x_values, y_values, **scatter_kws)


@with_active_style
def box_and_bars(
    box_values: Sequence[Iterable[float]],
    bar_values: Sequence[float],
    group_names: Sequence[str],
    colors: Optional[Sequence[str]] = None,
    edge_colors: Optional[Sequence[str]] = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
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
        edge_colors: Sequence of hex color codes for each group that is used for the
            edges of the boxes and bars. Must be the same length as group names. If
            None, black is used as edge color.

    Raises:
        ValueError: If the length of box_values, bar_values and group_names is not the
            same or if the length of colors is not the same as group_names.

    Returns:
        A matplotlib Figure and a list of Axes objects containing the box and bar plots.
    """
    if not (len(box_values) == len(bar_values) == len(group_names)):
        raise ValueError(
            "The length of 'box_values', 'bar_values' and 'group_names' must be the "
            "same."
        )
    if colors is not None and len(colors) != len(group_names):
        raise ValueError(
            "The length of 'colors' must be the same as the length of 'group_names'."
        )
    if edge_colors is not None and len(edge_colors) != len(group_names):
        raise ValueError(
            "The length of 'edge_colors' must be the same as the length of "
            "'group_names'."
        )

    if colors is None:
        colors = ["#D0D0D0" for _ in group_names]
    if edge_colors is None:
        edge_colors = ["#000000" for _ in group_names]

    num_samples = len(group_names)
    x_values = range(num_samples)
    bar_width = 0.8

    suptitle_space_inch = 0.4
    ax_height_inch = 1.6
    ax_hspace_inch = 0.35
    bar_width_inches = 0.24
    x_padding = 0.24
    fig_height = suptitle_space_inch + ax_height_inch * 2 + ax_hspace_inch

    fig_width = (num_samples + (2 * x_padding)) * bar_width_inches
    fig_size = (fig_width, fig_height)

    subplot_top = 1 - (suptitle_space_inch / fig_height)
    subplot_hspace = ax_hspace_inch / ax_height_inch

    bar_half_width = 0.5
    lower_xbound = (0 - bar_half_width) - x_padding
    upper_xbound = (num_samples - 1) + bar_half_width + x_padding

    fig, axes = plt.subplots(2, figsize=fig_size, sharex=True)
    fig.subplots_adjust(
        bottom=0, top=subplot_top, left=0, right=1, hspace=subplot_hspace
    )
    fig.suptitle("A box and bars plot", y=1)

    # Plot boxplots using the box_values
    ax = axes[0]
    ax.axhline(0, color="#999999", lw=1, zorder=2)
    boxplots = ax.boxplot(
        box_values,
        positions=x_values,
        vert=True,
        showfliers=False,
        patch_artist=True,
        widths=bar_width,
        medianprops={"color": "#000000"},
    )
    for color, edge_color, box in zip(
        colors, edge_colors, boxplots["boxes"], strict=True
    ):
        box.set(facecolor=color)
        box.set(edgecolor=edge_color)
    ylim = ax.get_ylim()
    ax.set_ylim(min(-0.4, ylim[0]), max(0.401, ylim[1]))

    # Plot barplots using the bar_values
    ax = axes[1]
    ax.bar(x_values, bar_values, width=bar_width, color=colors, edgecolor=edge_colors)
    ax.set_xticklabels(
        group_names, fontsize=plt.rcParams["axes.labelsize"], rotation=90
    )
    for ax in axes:
        ax.grid(False, axis="x")
    sns.despine(top=True, right=True)

    ax.set_xlim(lower_xbound, upper_xbound)
    return fig, axes
