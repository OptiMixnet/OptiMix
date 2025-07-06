
"""
This class or set of functions provides all the necessary code for generating the plots used in OptiMix.
"""




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke


def compatible(A, B):
    a = len(A)
    b = len(B)
    X = []

    Min = min(a, b)
    if not a > Min:
        i = 0
        for item in A:
            X.append(B[i])
            i = i + 1
        return A, X
    else:
        i = 0
        for item in B:
            X.append(A[i])
            i = i + 1
        return X, B

def CDF_Probability(data, T):
    data = np.array(data)
    return (1 - np.sum(data >= T) / data.size)

class LegendHandler(Line2D):
    def __init__(self, color, label, hatch_pattern):
        super().__init__([0], [0], color=color, lw=4, label=label)
        self.hatch_pattern = hatch_pattern

    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        line = super().create_artists(legend, orig_handle,
                                      x0, y0, width, height, fontsize, trans)
        for l in line:
            l.set_hatch(self.hatch_pattern)
        return line

class PLOT(object):
    def __init__(self, X, Y, Descriptions, X_label, Y_label, name, condition=False):
        self.cl1 = 'gray'
        self.cl2 = 'black'
        self.X = X
        self.Y = Y
        self.Desc = Descriptions
        self.XL = X_label
        self.YL = Y_label
        self.name = name
        self.condition = condition
        self.markers = ['D', 'v', '^', 's', '*', 'h', 'd']
        self.Line_style = ['-', ':', '--', '-.', '--']
        # Using a set of visually appealing colors
        self.colors = ['r','fuchsia','blue', 'cyan','green','orchid' ,'indigo', 'magenta', 'orange', 'yellow', 'lime', 'gold', 'seagreen']
        self.h = ['','/','.','x']
        self.Place = 'upper left'
    def scatter_line(self, Grid, y, Log=False):
        plt.close('all')

        legend_elements = []
        for i in range(len(self.Desc)):
                legend_elements.append(Line2D([0], [0],marker=self.markers[i],color=self.colors[i],
           lw=4,
           label=self.Desc[i],
           linestyle=self.Line_style[i],
           markersize=12)  # <-- This makes the marker bigger
)

        fig, ax = plt.subplots()
        ax.legend(handles=legend_elements, fontsize=14, loc= self.Place)

        if Log:
            plt.xscale("log")
        if Grid:
            plt.grid(linestyle='--', color=self.cl1, linewidth=1.5)

        for j in range(len(self.Y)):
            plt.plot(self.X, self.Y[j], alpha=1, color=self.colors[j], linestyle=self.Line_style[j], linewidth=2.5)
            plt.scatter(self.X, self.Y[j],
                        marker=self.markers[j],
                        s=160,  # size in points^2 (so 200 is a pretty big marker)
                        linewidths=3,
                        alpha=1,
                        facecolors='none',
                        edgecolors=self.colors[j])

        plt.ylim(0, y)

        plt.xlabel(self.XL, fontdict={'color': self.cl2, 'size': 20}, fontsize=17, fontweight='bold')
        plt.ylabel(self.YL, fontdict={'color': self.cl2, 'size': 20}, fontsize=17, fontweight='bold')
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 2

        ax.spines['top'].set_color(self.cl1)
        ax.spines['bottom'].set_color(self.cl1)
        ax.spines['left'].set_color(self.cl1)
        ax.spines['right'].set_color(self.cl1)

        ax.tick_params(axis='x', colors=self.cl2, which='both', labelsize=15)
        ax.tick_params(axis='y', colors=self.cl2, which='both', labelsize=15)

        plt.xticks(fontsize=15, weight='bold')
        plt.yticks(weight='bold', fontsize=15)
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().xaxis.set_tick_params(width=3)
        plt.gca().yaxis.set_tick_params(width=1.5)
        plt.gca().xaxis.set_tick_params(length=6)
        plt.gca().yaxis.set_tick_params(length=6)

        # Set the frame to be a rectangle with rounded corners
        ax.set_frame_on(True)
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10
        plt.tight_layout(rect=[0, 0, 1, 1], pad=0.1)

        # Set the x-axis ticks to the specified values

        
        plt.savefig(self.name, format='png', dpi=600)















    def scatter_area(self, Grid, y, Log=False):
        plt.close('all')

        legend_elements = []
        for i in range(len(self.Desc)):
            legend_elements.append(Line2D([0], [0], marker='', color=self.colors[i], lw=5, label=self.Desc[i],
                                          linestyle=self.Line_style[i]))

        fig, ax = plt.subplots()
        ax.legend(handles=legend_elements, fontsize=14, loc='upper right')

        if Log:
            plt.xscale("log")
        if Grid:
            plt.grid(linestyle='--', color='darkblue', linewidth=1.5)

        area_values = []

        for j in range(len(self.Y)):
            plt.plot(self.X, self.Y[j], alpha=1, color=self.colors[j], linestyle=self.Line_style[j], linewidth=2.5)
            plt.scatter(self.X, self.Y[j], marker='h', linewidths=0.7, alpha=1, color=self.colors[j])

            area = np.trapz(self.Y[j], self.X)
            area_values.append(area)

            hatch_pattern = '/' if j == 0 else '\\'
            plt.fill_between(self.X, self.Y[j], alpha=0.5, color=self.colors[j], hatch=hatch_pattern, edgecolor='black')

        plt.ylim(0, y)

        plt.xlabel(self.XL, fontdict={'color': 'darkblue', 'size': 20}, fontsize=17, fontweight='bold')
        plt.ylabel(self.YL, fontdict={'color': 'darkblue', 'size': 20}, fontsize=17, fontweight='bold')
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 2

        ax.spines['top'].set_color('darkblue')
        ax.spines['bottom'].set_color('darkblue')
        ax.spines['left'].set_color('darkblue')
        ax.spines['right'].set_color('darkblue')

        ax.tick_params(axis='x', colors='darkblue', which='both', labelsize=15)
        ax.tick_params(axis='y', colors='darkblue', which='both', labelsize=15)

        plt.xticks(fontsize=15, weight='bold')
        plt.yticks(weight='bold', fontsize=15)
        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().xaxis.set_tick_params(width=1.5)
        plt.gca().yaxis.set_tick_params(width=1.5)
        plt.gca().xaxis.set_tick_params(length=6)
        plt.gca().yaxis.set_tick_params(length=6)

        # Set the frame to be a rectangle with rounded corners
        ax.set_frame_on(True)
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10
        plt.tight_layout(rect=[0, 0, 1, 1], pad=0.1)

        plt.savefig(self.name, format='png', dpi=600)

        return area_values


    
    
  
    def Box_Plot(self,colors, y_max=None):

        self.colors = colors        
        """
        Creates a grouped box plot where each group is associated with a specific X value,
        and each group contains box plots for Y1i, Y2i, ...
        """
        plt.figure(figsize=(10, 5))

        num_groups = len(self.X)  # Number of x-axis positions
        num_categories = len(self.Y)  # Number of categories (Y1, Y2, ...)
        group_width = 0.6  # Total width for each group on the x-axis
        category_width = group_width / num_categories  # Width for each boxplot within a group

        for i in range(len(self.X)):  # Iterate over each x position
            positions = [self.X[i]*4 + j * category_width for j in range(num_categories)]
            data = [self.Y[j][i] for j in range(num_categories)]  # Get all Yij values for x_i
            
            for j in range(num_categories):
                # Assign different colors for Y1 and Y2 elements
                edge_color = self.colors[j % len(self.colors)]
                
                # Create box plots for each category at each X position
                plt.boxplot(data[j], positions=[positions[j]], widths=category_width * 0.9,
                            patch_artist=True,  # Enable custom colors
                            boxprops=dict(facecolor='white', edgecolor=edge_color, linewidth=3),
                            medianprops=dict(color='black', linewidth=3),
                            whiskerprops=dict(color=edge_color, linewidth=3),
                            capprops=dict(color=edge_color, linewidth=3),
                            flierprops=dict(marker='o', color='gray', alpha=0.06))

        # Set x-axis and y-axis labels
        plt.xlabel(self.XL, fontsize=30, fontweight='bold')
        plt.ylabel(self.YL, fontsize=30, fontweight='bold')

        # Add a legend for each category
        legend_elements = [
            plt.Line2D([0], [0], color=self.colors[i % len(self.colors)], lw=4, label=desc)
            for i, desc in enumerate(self.Desc)
        ]
        plt.legend(handles=legend_elements, fontsize=20, loc=self.Place, frameon=True)

        # Set x-axis tick labels
        plt.xticks([self.X[i]*4.02 + (num_categories - 1) * category_width / 2 for i in range(len(self.X))],
                   [x for x in ((self.X))], fontsize=25)


        # Set y-axis limit if y_max is provided
        if y_max is not None:
            plt.ylim(0,y_max)
        plt.tick_params(axis='y', labelsize=25)

        # Add grid
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the plot
        plt.tight_layout()
        plt.savefig(self.name, format='png', dpi=600)
        plt.show()

