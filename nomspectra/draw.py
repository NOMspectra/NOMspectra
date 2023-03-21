#    Copyright 2022 Volikov Alexander <ab.volikov@gmail.com>
#
#    This file is part of nomspectra. 
#
#    nomspectra is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    nomspectra is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with nomspectra.  If not, see <http://www.gnu.org/licenses/>.

from typing import Optional, Sequence, Tuple, Callable, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from .spectrum import Spectrum

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


def spectrum(spec: 'Spectrum',
    xlim: Tuple[Optional[float], Optional[float]] = (None, None),
    ylim: Tuple[Optional[float], Optional[float]] = (None, None),
    color: str = 'black',
    ax: Optional[plt.axes] = None,
    title: Optional[Union[str, bool]] = None,
    **kwargs
    ) -> None:
    """
    Draw mass spectrum

    All parameters except spec is optional

    Parameters
    ----------
    spec: Spectrum object
        spectrum for plot
    xlim: Tuple (float, float)
        restrict for mass
    ylim: Tuple (float, float)
        restrict for intensity
    color: str
        color of draw. Default black.
    ax: matplotlib axes object
        send here ax to plot in your own condition
    title: str
        Title of plot. Default None - Take name from metadata and number of peaks.
    **kwargs: dict
        additinal parameters to matplotlib plot method
    """

    df = spec.copy().table

    mass = df['mass'].values
    if xlim[0] is None:
        xlim = (mass.min(), xlim[1])
    if xlim[1] is None:
        xlim = (xlim[0], mass.max())

    intensity = df['intensity'].values
    # filter first intensity and only after mass (because we will lose the information)
    intensity = intensity[(xlim[0] <= mass) & (mass <= xlim[1])]
    mass = mass[(xlim[0] <= mass) & (mass <= xlim[1])]

    # bas solution, probably it's needed to rewrite this piece
    M = np.zeros((len(mass), 3))
    M[:, 0] = mass
    M[:, 1] = mass
    M[:, 2] = mass
    M = M.reshape(-1)

    I = np.zeros((len(intensity), 3))
    I[:, 1] = intensity
    I = I.reshape(-1)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4), dpi=75)

    ax.plot(M, I, color=color, linewidth=0.2, **kwargs)
    ax.plot([xlim[0], xlim[1]], [0, 0], color=color, linewidth=0.2, **kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('m/z, Da')
    ax.set_ylabel('Intensity')

    if title is None:
        if 'name' in spec.metadata:
            title = f'{spec.metadata["name"]}, {len(spec.table)} peaks'
        else:
            title = f'{len(spec.table)} peaks'
    if title:
        plt.title(title)

    return

def scatter(spec: 'Spectrum',
            x: str, 
            y: str,
            xlim: Tuple[Optional[float], Optional[float]] = (None, None),
            ylim: Tuple[Optional[float], Optional[float]] = (None, None),
            volume: str = 'intensity',
            color: Optional[str] = None, 
            alpha: float = 0.3, 
            size: Optional[float] = None,
            size_power: Optional[float] = None,
            ax: Optional[plt.axes] = None,
            title: Optional[Union[str, bool]] = None,
            **kwargs: Optional[dict]) -> None:
    """
    Draw scatter of different columns in mass-spectrum

    Parameters
    ----------
    spec: Spectrum object
        spectrum for plot
    x: str
        Name for x ordiante - columns in spec table
    y: str
        Name for y ordinate - columns in spec table
    xlim: Tuple (float, float)
        restrict for mass
    ylim: Tuple (float, float)
        restrict for intensity
    volume: str
        Name for z ordinate - columns in spec table.
        If size is none. size of dots will calculate by median of it parameter
    ax: plt.axes
        Optional, external ax
    color: str
        Optional. default None. Color for scatter.
        if None - separate color: CHO as blue, CHON as orange, CHOS as green and CHONS
    alpha: float
        Optional, default 0.3. Alpha for scatter
    size: float
        Optional. default None - normalize by intensivity to median.
    size_power: float
        Optinal. default None - plot linear dependes for volume.
        raises volume values to a power. For increae size put values > 1, for decrease <1
    title: str
        Title of draw. Default None. Take name from metadata and number of peaks.
    **kwargs: dict
        additional parameters to scatter method
    """
    
    if x not in spec.table:
        raise Exception(f'Value {x} is not in spectrum table. Calculate it before')
    if y not in spec.table:
        raise Exception(f'Value {y} is not in spectrum table. Calculate it before')

    spec = spec.copy().drop_unassigned()

    if volume == 'None':
        if size is None:
            raise Exception("when wolume is 'None' there must be size value")
        s = size
    else:
        s = spec.table[volume]/spec.table[volume].median()
        if size_power is not None:
            s = np.power(s, size_power)
        if size is not None:
            s = s * size

    if color is None:
        spec.table['color'] = 'blue'
        if 'N' in spec.table:
            spec.table.loc[(spec.table['C'] > 0) & (spec.table['H'] > 0) &(spec.table['O'] > 0) & (spec.table['N'] > 0), 'color'] = 'orange'
        if 'S' in spec.table:
            spec.table.loc[(spec.table['C'] > 0) & (spec.table['H'] > 0) &(spec.table['O'] > 0) & (spec.table['N'] < 1) & (spec.table['S'] > 0), 'color'] = 'green'
            spec.table.loc[(spec.table['C'] > 0) & (spec.table['H'] > 0) &(spec.table['O'] > 0) & (spec.table['N'] > 0) & (spec.table['S'] > 0), 'color'] = 'red'
        c = spec.table['color']
    else:
        c = color

    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4), dpi=75)

    ax.scatter(x=spec.table[x], y=spec.table[y], c=c, alpha=alpha, s=s, **kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if title is None:
        if 'name' in spec.metadata:
            title = f'{spec.metadata["name"]}, {len(spec.drop_unassigned())} formulas'
        else:
            title = f'{len(spec.drop_unassigned())} formulas'
    if title:
        plt.title(title)

    return

def scatter_density(spec: 'Spectrum',
                    x: str, 
                    y: str,
                    xlim: Tuple[Optional[float], Optional[float]] = (None, None),
                    ylim: Tuple[Optional[float], Optional[float]] = (None, None),
                    volume: str = 'intensity',
                    color: Optional[str] = None, 
                    alpha: float = 0.3, 
                    size: Optional[float] = None,
                    size_power: Optional[float] = None,
                    ax: Optional[plt.axes] = None,
                    title: Optional[Union[str, bool]] = None,
                    **kwargs) -> None:
    """
    Plot scatter with density
    Same as joinplot in seaborn
    but you can use external axes

    Parameters
    ----------
    spec: Spectrum object
        spec for plot
    ax: list of 3 plt.ax
        Optional, default None. List of three axes: ax for scatter, and ax_x, ax_y for density plot
    x: str
        Name for x ordiante - columns in spec table
    y: str
        Name for y ordinate - columns in spec table
    xlim: Tuple (float, float)
        restrict for mass
    ylim: Tuple (float, float)
        restrict for intensity
    volume: str
        Name for z ordinate - columns in spec table.
        If size is none. size of dots will calculate by median of it parameter
    color: str
        Optional. default None. Color for scatter.
        if None - separate color: CHO as blue, CHON as orange, CHOS as green and CHONS
    alpha: float
        Optional, default 0.3. Alpha for scatter
    size: float
        Optional. default None - normalize by intensivity to median.
    size_power: float
        Optinal. default None - plot linear dependes for volume.
        raises volume values to a power. For increae size put values > 1, for decrease <1
    title: str
        Title of draw. Default None. Take name from metadata and number of peaks.
    **kwargs: dict
        additional parameters to scatter method
    """

    if x not in spec.table:
        raise Exception(f'Value {x} is not in spectrum table. Calculate it before')
    if y not in spec.table:
        raise Exception(f'Value {y} is not in spectrum table. Calculate it before')

    if ax is None:
        fig = plt.figure(figsize=(6,6), dpi=100)
        gs = GridSpec(4, 4)

        ax = fig.add_subplot(gs[1:4, 0:3])
        ax_x = fig.add_subplot(gs[0,0:3])
        ax_y = fig.add_subplot(gs[1:4, 3])
    else:
        ax, ax_x, ax_y = ax

    scatter(spec, x=x, y=y, xlim=xlim, ylim=ylim, volume=volume, color=color, alpha=alpha, size=size, size_power=size_power, ax=ax, title=False, **kwargs)
    
    density(spec, col=x, color=color, ax=ax_x, xlim=xlim, title=False)
    ax_x.set_axis_off()
    
    density(spec, col=y, color=color, ax=ax_y, ylim=ylim, vertical=True, title=False)
    ax_y.set_axis_off()

    if title is None:
        if 'name' in spec.metadata:
            title = f'{spec.metadata["name"]}, {len(spec.drop_unassigned())} formulas'
        else:
            title = f'{len(spec.drop_unassigned())} formulas'
    if title:
        plt.title(title)

    return

def density(spec: 'Spectrum',
            col: str,
            xlim: Tuple[Optional[float], Optional[float]] = (None, None),
            ylim: Tuple[Optional[float], Optional[float]] = (None, None),
            color: str = 'blue', 
            ax: Optional[plt.axes] = None,
            title: Optional[Union[str, bool]] = None,
            vertical: bool = False,
            **kwargs: Optional[dict]) -> None:
    """
    Draw KDE density for values

    Parameters
    ----------
    spec: Spectrum object
        spec for plot
    x: str
        Column name for draw density
    xlim: Tuple (float, float)
        restrict for mass
    ylim: Tuple (float, float)
        restrict for intensity
    color: str
        Optional, default blue. Color of density plot
    ax: plt.axes
        Optional. External axes.
    title: str
        Title of draw. Default None. Take name from metadata and number of peaks.
    vertical: bool
        Flip x to y for vertical plot
    **kwargs: Dict
        Additional arguments for plot
    """

    if col not in spec.table:
        raise Exception(f'Value {col} is not in spectrum table. Calculate it before')

    spec = spec.copy().drop_unassigned()
    total_int = spec.table['intensity'].sum()

    x = np.linspace(spec.table[col].min(), spec.table[col].max(), 100)        
    oc = np.array([])
    
    for i, el in enumerate(x[1:]):
        s = spec.table.loc[(spec.table[col] > x[i-1]) & (spec.table[col] <= el), 'intensity'].sum()
        coun = len(spec.table) * s/total_int
        m = (x[i-1] + x[i])/2
        oc = np.append(oc, [m]*int(coun))

    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4), dpi=75)
    
    if vertical:
        sns.kdeplot(y = oc, ax=ax, color=color, fill=True, alpha=0.1, bw_adjust=2, **kwargs)
    else:
        sns.kdeplot(x = oc, ax=ax, color=color, fill=True, alpha=0.1, bw_adjust=2, **kwargs)
    ax.set_xlabel(col)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if title is None:
        if 'name' in spec.metadata:
            title = f'{spec.metadata["name"]}, {len(spec.drop_unassigned())} formulas'
        else:
            title = f'{len(spec.drop_unassigned())} formulas'
    if title:
        plt.title(title)

    return

def density_2D(spec: 'Spectrum', 
                x: str, 
                y: str,
                xlim: Tuple[Optional[float], Optional[float]] = (None, None),
                ylim: Tuple[Optional[float], Optional[float]] = (None, None),
                cmap: str ="YlGnBu", 
                shade: bool = True,
                title: Optional[Union[str, bool]] = None,
                ax: Optional[plt.axes] = None, 
                **kwargs
                ) -> None:
    """
    Draw 2D KDE density

    All parameters is optional

    Parameters
    ----------
    spec: Spectrum object
        spec for plot
    x: str
        Name for x ordiante - columns in spec table
    y: str
        Name for y ordinate - columns in spec table
    xlim: Tuple (float, float)
        restrict for mass
    ylim: Tuple (float, float)
        restrict for intensity
    cmap: str
        color map
    ax: matplotlib ax
        external ax
    shade: bool
        show shade
    title: str
        Title of draw. Default None. Take name from metadata and number of peaks.
    **kwargs: Dict
        Additional arguments for plot
    """

    if x not in spec.table:
        raise Exception(f'Value {x} is not in spectrum table. Calculate it before')
    if y not in spec.table:
        raise Exception(f'Value {y} is not in spectrum table. Calculate it before')

    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4), dpi=75)

    sns.kdeplot(x=spec.table[x], y=spec.table[y], ax=ax, cmap=cmap, fill=shade, **kwargs)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if title is None:
        if 'name' in spec.metadata:
            title = f'{spec.metadata["name"]}, {len(spec.drop_unassigned())} formulas'
        else:
            title = f'{len(spec.drop_unassigned())} formulas'
    if title:
        plt.title(title)

    return

def vk(spec: "Spectrum",
       func: Optional[Callable] = None,
       ax: Optional[plt.axes] = None,
       title: Optional[Union[str, bool]] = None,
       *args: Optional[list],
       **kwargs: Optional[dict]) -> None:
    """
    Scatter Van-Krevelin diagramm (O/C vs H/C)

    Parameters
    ----------
    spec: Spectrum object
        Mass-spectrum
    func: function
        function for draw vank-krevelen, may be scatter, scatter_density, density_2D
    ax: matplotlyp axes object
        send here ax to plot in your own condition
    title: str
        Title of draw. Default None. Take name from metadata and number of peaks.
    *args: list
        arguments to send scatter function
    **kwargs: dict
        arguments to send scatter function    
    """

    if 'O/C' or 'H/C' not in spec.table:
        spec = spec.hc_oc()

    if func is None:
        func = scatter

    func(spec=spec, x='O/C', y='H/C', xlim=(0, 1), ylim=(0, 2.2), ax=ax, title=False, *args, **kwargs)

    if title is None:
        if 'name' in spec.metadata:
            title = f'{spec.metadata["name"]}, {len(spec.drop_unassigned())} formulas'
        else:
            title = f'{len(spec.drop_unassigned())} formulas'
    if title:
        plt.title(title)

    return

def show_error(spec:"Spectrum") -> None:
    """
    Plot relative error of assigned brutto formulas vs mass

    Parameters
    ----------
    spec: Spectrum object
        mass-spec for plotting relative error
    """

    if "rel_error" not in spec.table:
        spec = spec.calc_error()      

    fig, ax = plt.subplots(figsize=(4, 4), dpi=75)
    ax.scatter(spec.table['mass'], spec.table['rel_error'], s=0.1)
    ax.set_xlabel('m/z, Da')
    ax.set_ylabel('error, ppm')

def venn(spec1: "Spectrum", 
         spec2: "Spectrum", 
         spec3: Optional["Spectrum"] = None,
         labels: Optional[Sequence[str]] = None,
         ax: Optional[plt.axes] = None,
         title: Optional[Union[str, bool]] = None,
         **kwargs):
    """
    Draw venn diagramm

    Parameters
    ----------
    spec1: Spectrum object
        first spectrum
    spec2: Spectrum object
        second spectrum
    spec3: Spectrum object
        third spectrum. Optional.
    labels: [Sequence[str]
        lables for circles.
    ax: matplotlib axes object
        send here ax to plot in your own condition
    title: str
        Title of plot. Default None - Take name from metadata and number of peaks.
    **kwargs: dict
        additinal parameters to venn2 or venn3 plot method
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4), dpi=75)

    s1 = set(spec1.table['calc_mass'].dropna().values)
    s2 = set(spec2.table['calc_mass'].dropna().values)

    if spec3 is None:
        v = 2
    else:
        v = 3
        s3 = set(spec3.table['calc_mass'].dropna().values)

    if labels is None:
        labels = [f'spec{i}' for i in range(1, v+1)]

    if v == 2:
        venn2(subsets = [s1, s2], set_labels = labels, ax=ax)
    else:
        print(labels)
        venn3(subsets = [s1, s2, s3], set_labels = labels, ax=ax)

    if title is not None:
        ax.set_title(title)

if __name__ == '__main__':
    pass