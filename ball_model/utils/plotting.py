""" 
Utils for pleasant plots 

Copyright 2023 Max-Planck-Gesellschaft
Code author: Jan Achterhold, jan.achterhold@tuebingen.mpg.de
Embodied Vision Group, Max Planck Institute for Intelligent Systems, Tübingen

This source code is licensed under the MIT license found in the
LICENSE.md file in the root directory of this source tree or at
https://opensource.org/licenses/MIT.
"""
import numpy as np

from ball_model.data import IDEAL_TABLE


class MplUtils():

    @classmethod
    def set_style(cls, plt, style):

        cls.style = style

        if style == "thesis":
            SMALL_SIZE = 9
            MEDIUM_SIZE = 11
            BIGGER_SIZE = 11
            figwidth = 1.05 * 5.89
        elif style == "l4dc":
            SMALL_SIZE = 8
            MEDIUM_SIZE = 8
            BIGGER_SIZE = 8
            linewidth = 379.4175
            POINTS_PER_INCH = 72.27
            figwidth = linewidth / POINTS_PER_INCH
        else:
            raise ValueError



        cls.figwidth = figwidth

        plt.rc('font', family="serif", size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE, labelsize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=SMALL_SIZE, dpi=200)  # fontsize of the figure title
        plt.rc('text', usetex=True)
        plt.rc('pdf', fonttype=42)

        if style == "thesis":
            plt.rc(
                'text.latex',
                preamble='\\usepackage{amsmath} \n \\usepackage[scaled=.97,helvratio=.93,p,theoremfont]{newpxtext} \n \\usepackage[vvarbb,smallerops,bigdelims]{newpxmath}'
            )
        elif style == "l4dc":
            plt.rc(
                'text.latex',
                preamble='\\usepackage{amsmath}'
            )
        else:
            raise ValueError

    @classmethod
    def get_width(cls, fraction=1, ncols=1):
        spacing_mm = 5
        spacing_in = spacing_mm / 25.4
        figwidth = fraction * (cls.figwidth - (ncols - 1) * spacing_in)

        return figwidth


def setup_board_axis(ax):
    half_length_of_table = 1.37
    half_width_of_table = 0.7625

    table_x_range = [
        IDEAL_TABLE[0] - half_width_of_table,
        IDEAL_TABLE[0] + half_width_of_table,
    ]
    table_y_range = [
        IDEAL_TABLE[1] - half_length_of_table,
        IDEAL_TABLE[1] + half_length_of_table,
    ]
    ax.fill_between(
        np.array(
            [IDEAL_TABLE[0] - half_width_of_table, IDEAL_TABLE[0] + half_width_of_table]
        ),
        np.array(
            [
                IDEAL_TABLE[1] - half_length_of_table,
                IDEAL_TABLE[1] - half_length_of_table,
            ]
        ),
        np.array(
            [
                IDEAL_TABLE[1] + half_length_of_table,
                IDEAL_TABLE[1] + half_length_of_table,
            ]
        ),
        color="gray",
        alpha=0.2,
        label="table",
    )
    ax.set_xlim(np.array(table_x_range) + np.array([-0.5, 0.5]))
    ax.set_ylim(np.array(table_y_range) + np.array([-0.5, 0.5]))
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xticks([-0.7, 1])


def setup_trajectory_axes(ax_arr):

    ax_arr[0].axhline(y=IDEAL_TABLE[2], linestyle="--", color="gray", alpha=0.6)
    ax_arr[0].set_xlim([-0.1, 1.3])
    ax_arr[0].set_ylim([-0.5, 0.5])
    ax_arr[0].set_xlabel("time [s]")
    ax_arr[0].set_ylabel("z [m]")

    setup_board_axis(ax_arr[1])
