from math import lcm
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import field
from pydantic.dataclasses import dataclass

from typing import Callable, Any, Sequence, Optional, Union, Dict


@dataclass
class PlotArgs:
    title: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    legend: list[str] = field(default_factory=list)
    xlim: Optional[tuple[float, float]] = None
    ylim: Optional[tuple[float, float]] = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.legend is None:
            self.legend = []

    def __hash__(self):
        return hash(
            (
                self.title,
                self.xlabel,
                self.ylabel,
                str(self.legend),
                self.xlim,
                self.ylim,
            )
        )

    def add_to_axes(self, ax: plt.Axes):
        if self.title is not None:
            ax.set_title(self.title)
        if self.xlabel is not None:
            ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            ax.set_ylabel(self.ylabel)
        if self.legend:
            ax.legend(self.legend)
        if self.xlim is not None:
            ax.set_xlim(self.xlim)
        if self.ylim is not None:
            ax.set_ylim(self.ylim)


class Plot:
    def __init__(
        self,
        plot_fn: Callable[[plt.Axes], Any],
        title=None,
        xlabel=None,
        ylabel=None,
        legend=None,
        xlim=None,
        ylim=None,
        **kwargs,
    ):
        self.plot_fn = plot_fn
        # assert that plot_fn has an Axes argument
        assert (
            "ax" in plot_fn.__code__.co_varnames
        ), "plot_fn must have an 'ax' argument"

        self.pargs = PlotArgs(title, xlabel, ylabel, legend, xlim, ylim, kwargs)

    def plot(self, ax: plt.Axes | None = None):
        if ax is None:
            ax = plt.gca()
        self.plot_fn(ax=ax)

        self.pargs.add_to_axes(ax)

    def __hash__(self):
        # hash the code and name of the plot function
        code_hash = hash(self.plot_fn.__code__)
        name_hash = hash(self.plot_fn.__name__)
        pargs_hash = hash(self.pargs)

        return hash((code_hash, name_hash, pargs_hash))

    def __mul__(self, other: "Plot"):
        """Plot them on the same axes"""

        def plot_fn(ax):
            self.plot(ax)
            other.plot(ax)

        plot_fn.__name__ = f"{self.plot_fn.__name__} x {other.plot_fn.__name__}"

        new_pargs = {**self.pargs.__dict__, **other.pargs.__dict__}
        legend = self.pargs.legend + other.pargs.legend
        new_pargs["legend"] = legend
        kwargs = new_pargs.pop("kwargs")
        return Plot(plot_fn, **new_pargs, **kwargs)

    def __repr__(self):
        return f"{self.pargs.title}({self.plot_fn.__name__})"


class View:
    def __init__(
        self, plots: list[Plot] | dict[int, Plot], arrangement: np.ndarray | None = None
    ):
        if isinstance(plots, Sequence):
            plots = {hash(plot): plot for plot in plots}
        elif isinstance(plots, Plot):
            plots = {hash(plots): plots}
        else:
            assert isinstance(plots, dict), "plots must be a list, dict, or Plot"

        self.plots = plots

        if arrangement is None:
            arrangement = np.array(list(plots.keys()), dtype=np.int64).reshape(1, -1)

        self.arrangement = np.array(arrangement, dtype=np.int64)

    def plot(self, **fig_args):

        fig_args.setdefault("figsize", self._calculate_figsize())

        # sharey, sharex patch
        sharex = fig_args.pop("sharex", False)
        sharey = fig_args.pop("sharey", False)
        hide_shared_xticks = fig_args.pop("hide_shared_xticks", True)
        hide_shared_yticks = fig_args.pop("hide_shared_yticks", True)

        fig, axes = plt.subplot_mosaic(self.arrangement, **fig_args)
        for plot_hash, plot in self.plots.items():
            if plot_hash != 0:
                plot.plot(axes[plot_hash])

        # sharey, sharex patch
        self._share_axes(axes, sharex, sharey, hide_shared_xticks, hide_shared_yticks)

        fig.tight_layout()

        axes_arrangement = np.zeros(self.arrangement.shape, dtype=object)
        for plot_hash, plot in self.plots.items():
            axes_arrangement[self.arrangement == plot_hash] = axes[plot_hash]

        return fig, axes_arrangement

    def _calculate_figsize(self) -> tuple[float, float]:
        return (self.arrangement.shape[1] * 4 + 4, self.arrangement.shape[0] * 3 + 2)

    def _share_axes(
        self,
        axes: Dict[int, plt.Axes],
        sharex: Union[bool, str],
        sharey: Union[bool, str],
        hide_shared_xticks: bool,
        hide_shared_yticks: bool,
    ) -> None:
        valid_share_options = [True, False, "row", "col"]
        if sharex not in valid_share_options or sharey not in valid_share_options:
            raise ValueError(f"sharex and sharey must be one of {valid_share_options}")

        self._apply_sharing(axes, "x", sharex, hide_shared_xticks)
        self._apply_sharing(axes, "y", sharey, hide_shared_yticks)

    def _apply_sharing(
        self,
        axes: Dict[int, plt.Axes],
        axis: str,
        share_option: Union[bool, str],
        hide_shared_ticks: bool,
    ) -> None:
        if share_option in ["row", "col"]:
            arr = self.arrangement if share_option == "row" else self.arrangement.T
            for row in arr:
                ax_arr = np.array([axes[col] for col in row])
                self._set_share_axes(ax_arr, axis, hide_shared_ticks)
        elif share_option is True:
            self._set_share_axes(np.array(list(axes.values())), axis, hide_shared_ticks)
        else:
            return

    @staticmethod
    def _set_share_axes(axs: np.ndarray, axis: str, hide_shared_ticks: bool) -> None:
        target = axs.flat[0]
        for ax in axs.flat:
            target._shared_axes[axis].join(target, ax)
        if hide_shared_ticks:
            for ax in axs.flat:
                getattr(ax, f"_label_outer_{axis}axis")(skip_non_rectangular_axes=True)


    def reshape(self, *args, **kwargs):
        self.arrangement = self.arrangement.reshape(*args, **kwargs)
        return self
    
    @property
    def T(self) -> "View":
        return View(self.plots, self.arrangement.T)

    def __add__(self, other: 'View') -> 'View':
        """Plot the views side by side (horizontally)"""
        return self._combine_views(other, axis=1)
    
    def __radd__(self, other):
    # This allows sum() to work (the default start value is zero)
        if other == 0:
            return self
        return self.__add__(other)

    def __truediv__(self, other: 'View') -> 'View':
        """Plot the views on top of each other (vertically)"""
        return self._combine_views(other, axis=0)

    def _combine_views(self, other: 'View', axis: int) -> 'View':
        if axis not in [0, 1]:
            raise ValueError("Axis must be 0 (vertical) or 1 (horizontal)")

        shape_index = 1 - axis  # 0 for horizontal, 1 for vertical
        lcm_shape = lcm(self.arrangement.shape[shape_index], other.arrangement.shape[shape_index])

        # Expand the arrangements to match in the combining dimension
        arrangement1 = np.repeat(self.arrangement, lcm_shape // self.arrangement.shape[shape_index], axis=shape_index)
        arrangement2 = np.repeat(other.arrangement, lcm_shape // other.arrangement.shape[shape_index], axis=shape_index)

        # Combine the arrangements
        new_arrangement = np.concatenate([arrangement1, arrangement2], axis=axis)

        return View(self.plots | other.plots, new_arrangement)

    def __mul__(self, other: Union['View', int]) -> 'View':
        """Plot the views on the same axes"""
        if isinstance(other, View):
            return self._plot_on_same_axes(other)
        elif isinstance(other, int):
            scale_factor = other
            # rescale the arrangement by the integer
            new_arrangement = np.repeat(self.arrangement, scale_factor, axis=0)
            new_arrangement = np.repeat(new_arrangement, scale_factor, axis=1)
            return View(self.plots, new_arrangement)
        
    
    def _plot_on_same_axes(self, other: 'View') -> 'View':
        # assert there is only one plot in each view
        assert all(len(view.plots) == 1 for view in [self, other])
        return View(
            [list(self.plots.values())[0] * list(other.plots.values())[0]],
        )

    def _repr_html_(self):
        fig, axes = self.plot()
        return fig._repr_html_()

    def __repr__(self):
        plot_arr = np.zeros(self.arrangement.shape, dtype=object)
        for plot_hash, plot in self.plots.items():
            plot_arr[self.arrangement == plot_hash] = plot
        return str(plot_arr)

