import matplotlib.pyplot as plt
import numpy as np
import json

from src.evaluation.process_results import read_data_for_models
from src.data.load_dataset import load_dataset
from src.utils import create_path
from src.paths import EVALUATION_RESULTS_PATH, OUTPUT_TABLE_PATH
from src.config import OUTPUT_TABLES_MODEL_NAMES

def plot_mean_coefficients(models, mean_coefficients, x_header, groups, colors, outfile, x_min=None, x_max=None, header=None, mean_coefficients_2=None, x_header_2=None):

    models = models[::-1] # reverse
    mean_coefficients = mean_coefficients[::-1] # reverse
    groups = groups[::-1] # reverse

    if mean_coefficients_2 is not None:
        mean_coefficients_2 = mean_coefficients_2[::-1]

    # create a figure and axis object
    if mean_coefficients_2 is None:
        fig, ax = plt.subplots()
        axes = [ax]
    else:
        fig, (ax, ax2) = plt.subplots(1 , 2 , sharex=True, sharey=False)
        fig.tight_layout(w_pad=1.4, h_pad=1)
        fig.set_size_inches(8, 4)
        axes = [ax, ax2]

    # add the header
    if header is not None:
        if mean_coefficients_2 is None:
            fig.text(0.5, 0.92, header, ha='center', fontdict={'family': 'serif', 'size': 8, "weight" : "bold"})
        else:
            fig.text(0.5, 1, header, ha='center', fontdict={'family': 'serif', 'size': 8, "weight" : "bold"})

    # remove the box around the plot
    for axis in axes:
        axis.spines['top'].set_visible(False)
        axis.spines['right'].set_visible(False)
        axis.spines['left'].set_visible(False)

        # set the bottom spine as visible
        axis.spines['bottom'].set_visible(True)

    # create a scatter plot of the mean coefficients
    for i, (mean, group) in enumerate(zip(mean_coefficients, groups)):
        bar = ax.barh(y=i, height=0.8, width=mean, color=colors[group], alpha=1)

    if mean_coefficients_2 is not None:
        for i, (mean, group) in enumerate(zip(mean_coefficients_2, groups)):
            bar = ax2.barh(y=i, height=0.8, width=mean, color=colors[group], alpha=1)

    # set the y-axis tick locations and labels
    ytick_pos = np.arange(len(models))

    for axis in axes:
        axis.set_yticks(ytick_pos)
        axis.set_yticklabels(models)
        # remove the y-axis tick marks
        axis.tick_params(axis='y', length=0, width=0)

    # set the x-axis limits and tick marks
    if mean_coefficients_2 is not None:
        if x_max is None:
            x_max = np.max([np.max(mean_coefficients), np.max(mean_coefficients_2)])
        if x_min is None:
            x_min = np.min([np.min(mean_coefficients), np.min(mean_coefficients_2)])
    else:
        if x_max is None:
            x_max = np.max(mean_coefficients)
        if x_min is None:
            x_min = np.min(mean_coefficients)
    length = x_max - x_min
    x_max += 0.05*length
    x_min -= 0.025*length

    for idx, axis in enumerate(axes):

        if idx == 0 or x_header_2 is None:
            x_head = x_header
        else:
            x_head = x_header_2

        axis.set_xlim(x_min, x_max)
        # ax.set_xticks(np.arange(x_min, x_max, 0.1))

        # remove the y-axis label
        ax.set_ylabel('')

        # Define the font properties
        
        import matplotlib.font_manager as fm
        font = fm.FontProperties(family='serif', size=8)
        # set the axis labels font properties
        axis.set_xlabel(x_head, fontproperties=fm.FontProperties(family='serif', size=8))
        axis.set_ylabel('', fontproperties=font)
        axis.xaxis.set_label_coords(1 - 0.0078 * (len(x_head.split("$")[0]) + len(x_head.split("$")) - 1), -0.1)
        # set the font type of the x-axis tick labels
        for label in axis.get_xticklabels() :
            label.set_fontproperties(fm.FontProperties(family='serif', size=8))
        axis.set_yticklabels(models, fontproperties=font)

        # make only the top two y-labels bold
        ytick_labels = axis.get_yticklabels()
        for i in range(len(ytick_labels)):
            if i >= len(ytick_labels)-4:
                ytick_labels[i].set_weight('bold')
            else:
                ytick_labels[i].set_weight('normal')

    ymin, ymax = ax.get_ylim()
    # add horizontal lines extending across the plot
    ytick_pos = np.arange(len(models))
    # draw line from left to right of plot

    line_left_edge = -0.12
    if mean_coefficients_2 is not None:
        line_left_edge = -0.2

    for axis in axes:
        fig.add_artist(plt.Line2D([line_left_edge, 1], [(1.5-ymin)/(ymax-ymin), (1.5-ymin)/(ymax-ymin)], color='lightgrey', linewidth=1, transform=axis.transAxes))
        fig.add_artist(plt.Line2D([line_left_edge, 1], [(10.5-ymin)/(ymax-ymin), (10.5-ymin)/(ymax-ymin)], color='lightgrey', linewidth=1, transform=axis.transAxes))

    if mean_coefficients_2 is not None:

        text = fig.text(-0.02, (2+9+4/2 + 0)/ymax, "Discrete choice",
            transform=fig.transFigure, va='center', fontdict={'family': 'serif', 'size': 8, "weight" : "bold"}, rotation=90,
                multialignment='center',
                verticalalignment='center', 
                horizontalalignment='center')
        
        text = fig.text(-0.02, (2+9/2 + 0.5)/ymax, "Positive-only",
            transform=fig.transFigure, va='center', fontdict={'family': 'serif', 'size': 8}, rotation=90,
                multialignment='center',
                verticalalignment='center', 
                horizontalalignment='center')
        
    else:

        text = fig.text(-0.16, (2+9+4/2 + 0)/ymax, "Discrete choice",
            transform=ax.transAxes, va='center', fontdict={'family': 'serif', 'size': 8, "weight" : "bold"}, rotation=90,
                multialignment='center',
                verticalalignment='center', 
                horizontalalignment='center')
        
        text = fig.text(-0.16, (2+9/2 + 0.5)/ymax, "Positive-only",
            transform=ax.transAxes, va='center', fontdict={'family': 'serif', 'size': 8}, rotation=90,
                multialignment='center',
                verticalalignment='center', 
                horizontalalignment='center')
        

    # save the plot to file and show it
    # plt.show()
    create_path(outfile)
    plt.savefig(outfile, format='svg', dpi=300, bbox_inches='tight')
    pass

def create_plots():
    """Create plots."""
    print("Creating plots...")

    with open(EVALUATION_RESULTS_PATH) as f:
        processed_results = json.load(f)

    raw_results = read_data_for_models()

    models = [model for model in OUTPUT_TABLES_MODEL_NAMES if model in processed_results["bias"]["overexposure"].keys()]

    # assign models to groups
    groups = ["Group 1"] * 4 + ["Group 2"] * 9 + ["Group 3"] * 2 
    # assign colors to groups
    colors = {'Group 1': '#203864', 'Group 2': '#4472C4', 'Group 3': 'gray'}

    ##### BIAS PLOTS ########
    plot_mean_coefficients(
        models=[OUTPUT_TABLES_MODEL_NAMES[model] for model in models],
        mean_coefficients=[np.mean(processed_results["bias"]["overexposure"][model]['ave_bias_coeffs_per_item']) for model in models],
        x_header="Mean bias $\hat{\delta}$",
        groups=groups,
        colors=colors,
        outfile="data/output/plots/bias_overexposure.svg",
        header="Exposure bias through overexposure",
    )

    plot_mean_coefficients(
        models=[OUTPUT_TABLES_MODEL_NAMES[model] for model in models],
        mean_coefficients=[np.mean(processed_results["bias"]["overexposure"][model]['ave_bias_coeffs_per_item']) for model in models],
        mean_coefficients_2=[np.mean(processed_results["bias_adjusted"][model]["ave_adj_bias_coeffs"]) for model in models],
        x_header="Mean bias $\hat{\delta}$",
        x_header_2="Adjusted mean bias $\hat{\delta}^'$",
        groups=groups,
        colors=colors,
        outfile="data/output/plots/bias_overexposure_side_by_side.svg",
        header="Exposure bias through overexposure",
    )

    plot_mean_coefficients(
        models=[OUTPUT_TABLES_MODEL_NAMES[model] for model in models],
        mean_coefficients=[np.mean(processed_results["bias_adjusted"][model]["ave_adj_bias_coeffs"]) for model in models],
        x_header="Adjusted mean bias $\hat{\delta}^'$",
        groups=groups,
        colors=colors,
        outfile="data/output/plots/bias_overexposure_adjusted.svg",
        header="Exposure bias through overexposure (adjusted)",
    )

    plot_mean_coefficients(
        models=[OUTPUT_TABLES_MODEL_NAMES[model] for model in models],
        mean_coefficients=[np.mean(processed_results["bias"]["competition"][model]['ave_bias_coeffs_per_item']) for model in models],
        x_header="Mean bias $\hat{\delta}$",
        groups=groups,
        colors=colors,
        outfile="data/output/plots/bias_competition.svg",
        header="Exposure bias through competition",
    )

    ##### PERFORMANCE PLOTS ########

    plot_mean_coefficients(
        models=[OUTPUT_TABLES_MODEL_NAMES[model] for model in models],
        mean_coefficients=[np.mean(raw_results["overexposure"][model]['mean_nDCG_B']) for model in models],
        x_header="nDCG",
        groups=groups,
        colors=colors,
        outfile="data/output/plots/performance_overexposure_B.svg",
        x_min=0.59,
        x_max=0.73,
        header="Mean nDCG for uniform frequencies$",
    )

    plot_mean_coefficients(
        models=[OUTPUT_TABLES_MODEL_NAMES[model] for model in models],
        mean_coefficients=[np.mean(raw_results["overexposure"][model]['mean_nDCG_BIAS']) for model in models],
        x_header="nDCG",
        groups=groups,
        colors=colors,
        outfile="data/output/plots/performance_overexposure_BIAS.svg",
        x_min=0.59,
        x_max=0.73,
        header="Mean nDCG for non-uniform exposure frequencies$",
    )

    plot_mean_coefficients(
        models=[OUTPUT_TABLES_MODEL_NAMES[model] for model in models],
        mean_coefficients=[np.mean(raw_results["overexposure"][model]['mean_nDCG_B']) for model in models],
        mean_coefficients_2=[np.mean(raw_results["overexposure"][model]['mean_nDCG_BIAS']) for model in models],
        x_header="nDCG",
        groups=groups,
        colors=colors,
        outfile="data/output/plots/performance_overexposure_side_by_side.svg",
        x_min=0.59,
        x_max=0.73,
        header="Mean nDCG for uniform and non-uniform exposure frequencies",
    )

    plot_mean_coefficients(
        models=[OUTPUT_TABLES_MODEL_NAMES[model] for model in models],
        mean_coefficients=[np.mean(raw_results["competition"][model]['mean_nDCG_popular']) for model in models],
        x_header="nDCG",
        groups=groups,
        colors=colors,
        outfile="data/output/plots/performance_competition_popular.svg",
        x_min=0.59,
        x_max=0.73,
        header="Mean nDCG for exposure with popular competitors",
    )

    plot_mean_coefficients(
        models=[OUTPUT_TABLES_MODEL_NAMES[model] for model in models],
        mean_coefficients=[np.mean(raw_results["competition"][model]['mean_nDCG_unpopular']) for model in models],
        x_header="nDCG",
        groups=groups,
        colors=colors,
        outfile="data/output/plots/performance_competition_unpopular.svg",
        x_min=0.59,
        x_max=0.73,
        header="Mean nDCG for exposure with unpopular competitors",
    )

    plot_mean_coefficients(
        models=[OUTPUT_TABLES_MODEL_NAMES[model] for model in models],
        mean_coefficients=[np.mean(raw_results["competition"][model]['mean_nDCG_popular']) for model in models],
        mean_coefficients_2=[np.mean(raw_results["competition"][model]['mean_nDCG_unpopular']) for model in models],
        x_header="nDCG",
        groups=groups,
        colors=colors,
        outfile="data/output/plots/performance_competition_side_by_side.svg",
        x_min=0.59,
        x_max=0.73,
        header="Mean nDCG for exposure with popular and unpopular competitors",
    )

    # table = generate_and_print_performance_table(
    #     title="Performance Overexposure",
    #     header=performance_header,
    #     models=models,
    #     n_tests=performance_n_tests,
    #     NDCG_data_unbiased={model: raw_results["overexposure"][model]["mean_nDCG_B"] for model in models},
    #     NDCG_data_biased={model: raw_results["overexposure"][model]["mean_nDCG_BIAS"] for model in models},
    #     pval_NDCGs_unbiased={model: processed_results["performance"]["overexposure"][model]["pval_nDCG_B"] for model in models},
    #     pval_NDCGs_biased={model: processed_results["performance"]["overexposure"][model]["pval_nDCG_BIAS"] for model in models},
    # )