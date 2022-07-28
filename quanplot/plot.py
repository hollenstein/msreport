from collections.abc import Iterable
from collections import UserDict

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import helper
import quantable
import quanalysis


class ColorWheelDict(UserDict):
    def __init__(self, colors: list[str] = None):
        self.data = {}

        if colors is not None:
            self.colors = colors
        else:
            self.colors = [
                '#80b1d3', '#fdb462', '#8dd3c7', '#bebada', '#fb8072',
                '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5'
            ]
        self._color_wheel = self.colors.copy()

    def _next_color(self) -> str:
        color = self._color_wheel.pop(0)
        self._color_wheel.append(color)
        return color

    def __setitem__(self, key, item):
        self.data[key] = item

    def __getitem__(self, key):
        if key not in self.data:
            self.data[key] = self._next_color()
        return self.data[key]


def missing_values_vertical(qtable: quantable.Qtable):
    """ Plot to analyze the completeness of protein quantification.

    If the column 'Valid' is present the data is filtered accordingly.
    Must be executed before imputation of missing values.
    """
    experiments = qtable.get_experiments()
    num_experiments = len(experiments)

    missing_values = quanalysis.count_missing_values(qtable)
    if 'Valid' in qtable.data:
        missing_values = missing_values[qtable.data['Valid']]

    barwidth = 0.8
    barcolors = ['#31A590', '#FAB74E', '#EB3952']
    figwidth = (num_experiments * 1.2) + 0.5
    figsize = (figwidth, 3.5)
    xtick_labels = ['No missing', 'Some missing', 'All missing']

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(1, num_experiments, figsize=figsize, sharey=True)
    for exp_num, exp in enumerate(experiments):
        ax = axes[exp_num]
        num_replicates = len(qtable.get_samples(exp))

        exp_missing = missing_values[f'Missing {exp}']
        missing_none = (exp_missing == 0).sum()
        missing_some = ((exp_missing > 0) & (exp_missing < num_replicates)).sum()
        missing_all = (exp_missing == 3).sum()

        y = [missing_none, missing_some, missing_all]
        x = range(len(y))
        ax.bar(x, y, width=barwidth, color=barcolors)
        if exp_num == 0:
            ax.set_ylabel('# Proteins')
        ax.set_title(exp)
        ax.set_xticks(np.array([0, 1, 2]) + 0.4)
        ax.set_xticklabels(xtick_labels, rotation=45, va='top', ha='right')
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#000000')
            ax.spines[spine].set_linewidth(1)
        ax.grid(False, axis='x')
        ax.grid(axis='y', linestyle='dashed', linewidth=1)
    sns.despine(top=True, right=True)
    fig.tight_layout()
    return fig, axes


def missing_values_horizontal(qtable: quantable.Qtable):
    """ Plot to analyze the completeness of protein quantification.

    If the column 'Valid' is present the data is filtered accordingly.
    Must be executed before imputation of missing values.
    """
    experiments = qtable.get_experiments()
    num_experiments = len(experiments)

    missing_values = quanalysis.count_missing_values(qtable)
    if 'Valid' in qtable.data:
        missing_values = missing_values[qtable.data['Valid']]

    data = {'exp': [], 'max': [], 'some': [], 'min': []}
    for exp in experiments:
        exp_missing = missing_values[f'Missing {exp}']
        total = len(exp_missing)
        num_replicates = len(qtable.get_samples(exp))
        missing_all = (exp_missing == num_replicates).sum()
        missing_none = (exp_missing == 0).sum()
        with_missing_some = total - missing_all

        data['exp'].append(exp)
        data['max'].append(total)
        data['some'].append(with_missing_some)
        data['min'].append(missing_none)

    plotheight = (num_experiments * 0.5) + 0.5
    legendheight = 1.5
    figheight = plotheight + legendheight
    figsize = (5, figheight)

    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(y='exp', x='max', data=data,
                label='All missing', color='#EB3952')
    sns.barplot(y='exp', x='some', data=data,
                label='Some missing', color='#FAB74E')
    sns.barplot(y='exp', x='min', data=data,
                label='None missing', color='#31A590')

    ax.set_xlim(0, total)
    ax.set_title('Completeness of protein quantification per experiment')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, bbox_to_anchor=(1, 0), ncol=3
    )
    figure_space_for_legend = 1 - (legendheight / figheight)
    fig.tight_layout(rect=[0, 0, 1, figure_space_for_legend])
    return fig, ax


def sample_intensities(qtable: quantable.Qtable, tag: str = 'Intensity',
                       is_log: bool = False) -> (plt.Figure, plt.Axes):
    """ Plot to analyze similarity of sample intensities. """
    columns = helper.find_columns(qtable.data, tag, must_be_substring=True)
    matrix = qtable.data[columns].copy()
    matrix.columns = [c.replace(tag, '').strip() for c in columns]
    matrix = matrix.replace({0: np.nan})

    if is_log:
        log2_matrix = matrix
        matrix = np.power(2, log2_matrix)
    else:
        log2_matrix = np.log2(matrix)
    samples = matrix.columns.tolist()

    finite_values = (log2_matrix.isna().sum(axis=1) == 0)
    pseudo_ref = np.nanmean(log2_matrix[finite_values], axis=1)
    log2_ratios = log2_matrix[finite_values].subtract(pseudo_ref, axis=0)

    bar_values = matrix.sum()
    box_values = [log2_ratios[c] for c in log2_ratios.columns]
    color_wheel = ColorWheelDict()
    colors = [color_wheel[exp] for exp in qtable.get_experiments(samples)]
    fig, axes = box_and_bars(box_values, bar_values, samples, colors=colors)
    axes[0].set_title(f'Comparison of "{tag}" columns', pad=15)
    fig.tight_layout()
    return fig, axes


def box_and_bars(box_values: Iterable[Iterable[float]],
                 bar_values: Iterable[float], sample_names: list[str],
                 colors: list[str] = None) -> (plt.Figure, plt.Axes):
    assert len(box_values) == len(bar_values) == len(sample_names)
    num_samples = len(sample_names)
    x_values = range(num_samples)
    width = 0.8
    xlim = (-1 + 0.15, num_samples - 0.15)
    figwidth = (num_samples * 0.25) + 1.1
    figsize = (figwidth, 6)

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, figsize=figsize, sharex=True)

    # Plot boxplots of sample to pseudo reference ratios
    ax = axes[0]
    ax.plot(xlim, (0, 0), color='#999999', lw=1, zorder=2)
    boxplots = ax.boxplot(
        box_values, positions=x_values, vert=True, showfliers=False,
        patch_artist=True, widths=width, medianprops={'color': '#000000'}
    )
    for color, box in zip(colors, boxplots['boxes']):
        box.set(facecolor=color)
    ylim = ax.get_ylim()
    ax.set_ylim(min(-0.4, ylim[0]), max(0.401, ylim[1]))
    ax.set_ylabel('Protein ratios [log2]\nto pseudo reference')

    # Plot total sample intensities
    ax = axes[1]
    ax.bar(x_values, bar_values, width=width, color=colors, edgecolor='#000000')
    ax.set_ylabel('Total protein intensity')
    ax.set_xticklabels(sample_names, rotation=90)

    for ax_pos, ax in enumerate(axes):
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#000000')
            ax.spines[spine].set_linewidth(1)
        ax.grid(False, axis='x')
        ax.grid(axis='y', linestyle='dashed', linewidth=1, color='#cccccc')
    sns.despine(top=True, right=True)

    ax.set_xlim(xlim)
    fig.tight_layout()
    return fig, axes
