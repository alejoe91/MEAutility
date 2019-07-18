from __future__ import print_function

import numpy as np
from MEAutility.core import MEA, RectMEA, rotation_matrix
import matplotlib.pylab as plt

import matplotlib.patches as patches
from matplotlib.path import Path
from mpl_toolkits.mplot3d import art3d
from matplotlib import colors as mpl_colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_probe(mea, ax=None, xlim=None, ylim=None, color_currents=False, top=None, bottom=None,
               cmap='viridis', type='shank', alpha_elec=0.7, alpha_prb=0.3):
    '''
    Plots probe in 2d.
    
    Parameters
    ----------
    mea: MEA object
        The MEA to be plotted
    ax: matplotlib axis
        The axis to plot on
    xlim: list
        Limits for x axis
    ylim: list
        Limits for y axis
    color_currents: bool
        If True currents are color-coded
    top: int
        The length of the probe in the top direction
    bottom: int
        The length of the probe in the bottom direction
    cmap: matplotlib colormap
        The colormap to use for currents
    type: str
        'shank' or 'plane' type
    alpha_elec: float
        Alpha value for electrodes
    alpha_prb: float
        Alpha value for probe

    Returns
    -------
    ax: matplotlib axis
        The output axis

    '''

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    n_elec = mea.positions.shape[0]
    elec_size = mea.size

    if mea.type == 'mea':
        mea_pos = np.array([np.dot(mea.positions, mea.main_axes[0]), np.dot(mea.positions, mea.main_axes[1])]).T

        min_x, max_x = [np.min(np.dot(mea.positions, mea.main_axes[0])),
                        np.max(np.dot(mea.positions, mea.main_axes[0]))]
        center_x = (min_x + max_x) / 2.
        min_y, max_y = [np.min(np.dot(mea.positions, mea.main_axes[1])),
                        np.max(np.dot(mea.positions, mea.main_axes[1]))]

        if type == 'shank':
            if top is None:
                probe_height = 200
            else:
                probe_height = top
            probe_top = max_y + probe_height
            if bottom is None:
                probe_bottom = min_y - probe_height
            else:
                probe_bottom = min_y - bottom
            probe_corner = min_y - 0.1 * probe_height
            probe_left = min_x - 0.1 * probe_height
            probe_right = max_x + 0.1 * probe_height

            verts = [
                (min_x - 2 * elec_size, probe_top),  # left, bottom
                (min_x - 2 * elec_size, probe_corner),  # left, top
                (center_x, probe_bottom),  # right, top
                (max_x + 2 * elec_size, probe_corner),  # right, bottom
                (max_x + 2 * elec_size, probe_top),
                (min_x - 2 * elec_size, max_y + 2 * elec_size)  # ignored
            ]

            codes = [Path.MOVETO,
                     Path.LINETO,
                     Path.LINETO,
                     Path.LINETO,
                     Path.LINETO,
                     Path.CLOSEPOLY,
                     ]

        elif type == 'planar':
            probe_top = max_y + 2 * elec_size
            probe_bottom = min_y - 2 * elec_size
            probe_left = min_x - 2 * elec_size
            probe_right = max_x + 2 * elec_size

            verts = [
                (min_x - 2 * elec_size, max_y + 2 * elec_size),  # left, bottom
                (min_x - 2 * elec_size, min_y - 2 * elec_size),  # left, top
                (max_x + 2 * elec_size, min_y - 2 * elec_size),  # right, bottom
                (max_x + 2 * elec_size, max_y + 2 * elec_size),  # ignored
                (max_x + 2 * elec_size, max_y + 2 * elec_size)  # ignored
            ]

            codes = [Path.MOVETO,
                     Path.LINETO,
                     Path.LINETO,
                     Path.LINETO,
                     Path.CLOSEPOLY,
                     ]
        else:
            raise AttributeError("'type' can be 'shank' or 'planar'")

        path = Path(verts, codes)

        patch = patches.PathPatch(path, facecolor='green', edgecolor='k', lw=0.5, alpha=alpha_prb)
        ax.add_patch(patch)

        if color_currents:
            norm_curr = mea.currents / np.max(np.abs(mea.currents))
            colormap = plt.get_cmap(cmap)
            elec_colors = colormap(norm_curr)
        else:
            elec_colors = ['orange'] * mea.number_electrodes

        if mea.shape == 'square':
            for e in range(n_elec):
                elec = patches.Rectangle((mea_pos[e, 0] - elec_size, mea_pos[e, 1] - elec_size), 2 * elec_size,
                                         2 * elec_size, alpha=alpha_elec, facecolor=elec_colors[e],
                                         edgecolor=[0.3, 0.3, 0.3], lw=0.5)

                ax.add_patch(elec)
        elif mea.shape == 'circle':
            for e in range(n_elec):
                elec = patches.Circle((mea_pos[e, 0], mea_pos[e, 1]), elec_size, alpha=alpha_elec,
                                      facecolor=elec_colors[e], edgecolor=[0.3, 0.3, 0.3], lw=0.5)

                ax.add_patch(elec)
    else:
        mea_pos = np.array([np.dot(mea.positions, mea.main_axes[0]), np.dot(mea.positions, mea.main_axes[1])]).T

        if top is None:
            top = 200
        for pos in mea_pos:
            ax.add_patch(plt.Rectangle((pos[0] - mea.size, 0), 2 * mea.size, top, facecolor='gray',
                                       alpha=alpha_prb, edgecolor='k'))
        probe_left = np.min(mea_pos)
        probe_right = np.max(mea_pos)
        probe_bottom = -50
        probe_top = top

    ax.axis('equal')

    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(probe_left - 5 * elec_size, probe_right + 5 * elec_size)
    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(probe_bottom - 5 * elec_size, probe_top + 5 * elec_size)

    return ax


def plot_probe_3d(mea, ax=None, xlim=None, ylim=None, zlim=None, top=None, bottom=None, type='shank',
                  color_currents=False, cmap='viridis', alpha_elec=0.7, alpha_prb=0.3):
    '''
    Plots probe in 3d.

    mea: MEA object
        The MEA to be plotted
    ax: matplotlib axis
        The axis to plot on
    xlim: list
        Limits for x axis
    ylim: list
        Limits for y axis
    ylim: list
        Limits for z axis
    color_currents: bool
        If True currents are color-coded
    top: int
        The length of the probe in the top direction
    bottom: int
        The length of the probe in the bottom direction
    cmap: matplotlib colormap
        The colormap to use for currents
    type: str
        'shank' or 'plane' type
    alpha_elec: float
        Alpha value for electrodes
    alpha_prb: float
        Alpha value for probe

    Returns
    -------
    ax: matplotlib axis
        The output axis

    '''
    from matplotlib import colors as mpl_colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    if color_currents:
        norm_curr = mea.currents / np.max(np.abs(mea.currents))
        norm_curr += np.abs(np.min(norm_curr))
        colormap = plt.get_cmap(cmap)
        elec_colors = colormap(norm_curr)
    else:
        elec_colors = ['orange'] * mea.number_electrodes

    if mea.type == 'mea':
        elec_list = []
        if mea.shape == 'square':
            for pos in mea.positions:
                elec = np.array([pos - mea.size * mea.main_axes[0] - mea.size * mea.main_axes[1],
                                 pos - mea.size * mea.main_axes[0] + mea.size * mea.main_axes[1],
                                 pos + mea.size * mea.main_axes[0] + mea.size * mea.main_axes[1],
                                 pos + mea.size * mea.main_axes[0] - mea.size * mea.main_axes[1]])
                elec_list.append(elec)
            el = Poly3DCollection(elec_list, alpha=0.8, color=elec_colors)
            el.set_alpha(alpha_elec)
            ax.add_collection3d(el)
        elif mea.shape == 'circle':
            for i, pos in enumerate(mea.positions):
                p = _make_3d_ellipse_patch(mea.size, mea.main_axes[0], mea.main_axes[1],
                                           pos, ax, facecolor=elec_colors[i], edgecolor=None, alpha=alpha_elec)
                p.set_alpha(alpha_elec)

        center_probe = mea.positions.mean(axis=0)
        min_dim_1c = np.min(np.dot(mea.positions - center_probe, mea.main_axes[0]))
        min_dim_2c = np.min(np.dot(mea.positions - center_probe, mea.main_axes[1]))
        max_dim_1c = np.max(np.dot(mea.positions - center_probe, mea.main_axes[0]))
        max_dim_2c = np.max(np.dot(mea.positions - center_probe, mea.main_axes[1]))
        minmin, maxmax = [2 * np.min([min_dim_1c, min_dim_2c]),
                          2 * np.max([max_dim_1c, max_dim_2c])]

        if type == 'shank':
            probe_height = 200
            if top is None:
                probe_top = (max_dim_2c + probe_height) * mea.main_axes[1]
            else:
                probe_top = top * mea.main_axes[1]
            if bottom is None:
                probe_bottom = (min_dim_2c - probe_height) * mea.main_axes[1]
            else:
                probe_bottom = (min_dim_2c - bottom) * mea.main_axes[1]
            probe_corner = (min_dim_2c - 0.1 * probe_height) * mea.main_axes[1]

            verts = np.array([
                (min_dim_1c - 2 * mea.size) * mea.main_axes[0] + probe_top + center_probe,  # left, bottom
                (min_dim_1c - 2 * mea.size) * mea.main_axes[0] + probe_corner + center_probe,  # left, top
                probe_bottom + center_probe,  # right, top
                (max_dim_1c + 2 * mea.size) * mea.main_axes[0] + probe_corner + center_probe,  # right, bottom
                (max_dim_1c + 2 * mea.size) * mea.main_axes[0] + probe_top + center_probe,
            ])

        elif type == 'planar':
            verts = np.array([
                (min_dim_1c - 3 * mea.size) * mea.main_axes[0] + (max_dim_2c + 3 * mea.size) * mea.main_axes[1]
                + center_probe,  # left, bottom
                (min_dim_1c - 3 * mea.size) * mea.main_axes[0] + (min_dim_2c - 3 * mea.size) * mea.main_axes[1]
                + center_probe,
                (max_dim_1c + 3 * mea.size) * mea.main_axes[0] + (min_dim_2c - 3 * mea.size) * mea.main_axes[1]
                + center_probe,
                (max_dim_1c + 3 * mea.size) * mea.main_axes[0] + (max_dim_2c + 3 * mea.size) * mea.main_axes[1]
                + center_probe,
            ])
        else:
            raise AttributeError("'type' can be 'planar' or 'shank'")

        r = Poly3DCollection([verts])
        alpha = (alpha_prb,)
        mea_col = mpl_colors.to_rgb('g') + alpha
        edge_col = mpl_colors.to_rgb('k') + alpha
        r.set_edgecolor(edge_col)
        r.set_facecolor(mea_col)
        ax.add_collection3d(r)

    else:
        if top is None:
            top = 200
        for pos in mea.positions:
            plot_cylinder_3d(pos, direction=-mea.normal, radius=mea.size, length=top, color='gray', alpha=alpha_prb,
                             ax=ax)

        center_probe = mea.positions.mean(axis=0)

        minmin = -top
        maxmax = top

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim([minmin, maxmax] + center_probe[0])
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([minmin, maxmax] + center_probe[1])
    if zlim is not None:
        ax.set_zlim(zlim)
    else:
        ax.set_zlim([minmin, maxmax] + center_probe[2])

    return ax


def plot_v_image(mea, v_plane=None, x_bound=None, y_bound=None, z_bound=None, offset=0,
                 plane='yz', npoints=30, ax=None, cmap='viridis', **kwargs):
    '''
    Plots voltage image generated by mea currents.
    
    Parameters
    ----------
    mea: MEA object
        The MEA to be plotted
    v_plane: np.array
        The voltage values to be plotted (default=None)
    x_bound: list
        X boundaries to compute grid (if plane has 'x' dimension)
    y_bound: list
        Y boundaries to compute grid (if plane has 'y' dimension)
    z_bound: list
        Z boundaries to compute grid (if plane has 'z' dimension) 
    offset: float
        Offset in um from probe plane to compute electrical potential
    plane: str
        'xy', 'yz', 'xz'
    npoints: int
        Number of points in grid
    ax: matplotlib axis
        The axis to plot on
    cmap: matplotlib colormap
        The colormap to use for currents
    kwargs: keyword args
        Other arguments for matshow function

    Returns
    -------
    ax: matplotlib axis
        The output axis
    v_grid: np.array
        The voltage image 

    '''
    if v_plane is None:
        v_grid = np.zeros((npoints, npoints))
        if plane == 'xy':
            assert x_bound is not None and y_bound is not None
            vec1 = np.linspace(x_bound[0], x_bound[1], npoints)
            vec2 = np.linspace(y_bound[0], y_bound[1], npoints)
            for i, v1 in enumerate(vec1):
                for j, v2 in enumerate(vec2):
                    v_grid[i, j] = mea.compute_field(np.array([v1, v2, offset]))
        elif plane == 'yz':
            assert y_bound is not None and z_bound is not None
            vec1 = np.linspace(y_bound[0], y_bound[1], npoints)
            vec2 = np.linspace(z_bound[0], z_bound[1], npoints)
            for i, v1 in enumerate(vec1):
                for j, v2 in enumerate(vec2):
                    v_grid[i, j] = mea.compute_field(np.array([offset, v1, v2]))
        elif plane == 'xz':
            assert x_bound is not None and z_bound is not None
            vec1 = np.linspace(x_bound[0], x_bound[1], npoints)
            vec2 = np.linspace(z_bound[0], z_bound[1], npoints)
            for i, v1 in enumerate(vec1):
                for j, v2 in enumerate(vec2):
                    v_grid[i, j] = mea.compute_field(np.array([v1, offset, v2]))

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.matshow(v_grid.T, origin='lower', extent=[vec1[0], vec1[-1], vec2[0], vec2[-1]], cmap=cmap, **kwargs)
    else:
        v_grid = v_plane.T
        if plane == 'xy':
            assert x_bound is not None and y_bound is not None
            vec1 = np.linspace(x_bound[0], x_bound[1], npoints)
            vec2 = np.linspace(y_bound[0], y_bound[1], npoints)
        elif plane == 'yz':
            assert y_bound is not None and z_bound is not None
            vec1 = np.linspace(y_bound[0], y_bound[1], npoints)
            vec2 = np.linspace(z_bound[0], z_bound[1], npoints)
        elif plane == 'xz':
            assert x_bound is not None and z_bound is not None
            vec1 = np.linspace(x_bound[0], x_bound[1], npoints)
            vec2 = np.linspace(z_bound[0], z_bound[1], npoints)
        ax.matshow(v_plane, origin='lower', extent=[vec1[0], vec1[-1], vec2[0], vec2[-1]], cmap=cmap, **kwargs)

    return ax, v_grid.T


def plot_v_surf(mea, v_plane=None, x_bound=None, y_bound=None, z_bound=None, offset=0,
                plane='yz', plot_plane=None, npoints=30, ax=None, cmap='viridis', alpha=0.8, distance=30):
    '''
    Plots voltage image generated by mea currents.
    
    Parameters
    ----------
    mea: MEA object
        The MEA to be plotted
    v_plane: np.array
        The voltage values to be plotted (default=None)
    x_bound: list
        X boundaries to compute grid (if plane has 'x' dimension)
    y_bound: list
        Y boundaries to compute grid (if plane has 'y' dimension)
    z_bound: list
        Z boundaries to compute grid (if plane has 'z' dimension) 
    offset: float
        Offset in um from probe plane to compute electrical potential
    plane: str
        'xy', 'yz', 'xz'
    plot_plane: str
        Plane to plot surf 
    npoints: int
        Number of points in grid
    ax: matplotlib axis
        The axis to plot on
    cmap: matplotlib colormap
        The colormap to use for currents
    alpha: float
        Aplha value for surf plot
    distance: float
        Distance of surf plot from mea
    
    Returns
    -------
    ax: matplotlib axis
        The output axis
    v_grid: np.array
        The voltage image 
    '''
    if v_plane is None:
        v_grid = np.zeros((npoints, npoints))
        if plane == 'xy':
            assert x_bound is not None and y_bound is not None
            vec1 = np.linspace(x_bound[0], x_bound[1], npoints)
            vec2 = np.linspace(y_bound[0], y_bound[1], npoints)
            for i, v1 in enumerate(vec1):
                for j, v2 in enumerate(vec2):
                    v_grid[i, j] = mea.compute_field(np.array([v1, v2, offset]))
        elif plane == 'yz':
            assert y_bound is not None and z_bound is not None
            vec1 = np.linspace(y_bound[0], y_bound[1], npoints)
            vec2 = np.linspace(z_bound[0], z_bound[1], npoints)
            for i, v1 in enumerate(vec1):
                for j, v2 in enumerate(vec2):
                    v_grid[i, j] = mea.compute_field(np.array([offset, v1, v2]))
        elif plane == 'xz':
            assert x_bound is not None and z_bound is not None
            vec1 = np.linspace(x_bound[0], x_bound[1], npoints)
            vec2 = np.linspace(z_bound[0], z_bound[1], npoints)
            for i, v1 in enumerate(vec1):
                for j, v2 in enumerate(vec2):
                    v_grid[i, j] = mea.compute_field(np.array([v1, offset, v2]))

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1, projection='3d')

        v1_plane, v2_plane = np.meshgrid(vec1, vec2)
        surf1 = ax.plot_surface(v1_plane,
                                v2_plane,
                                v_grid.T,
                                cmap=cmap,
                                alpha=alpha,
                                zorder=0,
                                antialiased=True)
        if plot_plane is not None:
            if plot_plane == 'xy':
                _rotate_poly3Dcollection(surf1, [1, 0, 0], [0, 1, 0], shift=[0, 0, distance])
            elif plot_plane == 'yz':
                _rotate_poly3Dcollection(surf1, [0, 1, 0], [0, 0, 1], shift=[distance, 0, 0])
            elif plot_plane == 'xz':
                _rotate_poly3Dcollection(surf1, [1, 0, 0], [0, 0, 1], shift=[0, distance, 0])
    else:
        v_grid = v_plane
        if plane == 'xy':
            assert x_bound is not None and y_bound is not None
            vec1 = np.linspace(x_bound[0], x_bound[1], npoints)
            vec2 = np.linspace(y_bound[0], y_bound[1], npoints)
        elif plane == 'yz':
            assert y_bound is not None and z_bound is not None
            vec1 = np.linspace(y_bound[0], y_bound[1], npoints)
            vec2 = np.linspace(z_bound[0], z_bound[1], npoints)
        elif plane == 'xz':
            assert x_bound is not None and z_bound is not None
            vec1 = np.linspace(x_bound[0], x_bound[1], npoints)
            vec2 = np.linspace(z_bound[0], z_bound[1], npoints)
        v1_plane, v2_plane = np.meshgrid(vec1, vec2)
        surf1 = ax.plot_surface(v1_plane,
                                v2_plane,
                                v_grid.T,
                                cmap=cmap,
                                alpha=alpha,
                                zorder=0,
                                antialiased=True)
        if plot_plane is not None:
            if plot_plane == 'xy':
                _rotate_poly3Dcollection(surf1, [1, 0, 0], [0, 1, 0], shift=[0, 0, distance])
            elif plot_plane == 'yz':
                _rotate_poly3Dcollection(surf1, [0, 1, 0], [0, 0, 1], shift=[distance, 0, 0])
            elif plot_plane == 'xz':
                _rotate_poly3Dcollection(surf1, [1, 0, 0], [0, 0, 1], shift=[0, distance, 0])

    return ax, v_grid.T


def plot_mea_recording(signals, mea, colors=None, lw=1, ax=None, spacing=None,
                       scalebar=False, time=None, vscale=None, hide_axis=True,
                       axis_equal=False):
    '''
    Plots mea signals at electrode locations.

    Parameters
    ----------
    signals: np.array
        The signals to plot. Can be 2D (single signals) or 3D (multiple signals)
    mea: MEA object
        The MEA to be plotted
    colors: matplotlib colors
        The color or colors to use
    lw: float
        Line width of the lines
    ax: matplotlib axis
        The axis to plot on
    spacing: float
        The spacing in the x-direction
    scalebar: bool
        If True, a scale bar is plotted
    time: float
        If scalebar is True, the time of the scalebar
    vscale: float
        The scale to use for the signal (default .5 times the maximum signal)
    hide_axis: bool
        If True (default), axis are hidden
    axis_equal: bool
        If True, axis aspect is set to 'equal'

    Returns
    -------
    ax: matplotlib axis
        The output axis

    '''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, frameon=False)
        no_tight = False
    else:
        fig = ax.get_figure()
        no_tight = True

    mea_pos = np.array([np.dot(mea.positions, mea.main_axes[0]), np.dot(mea.positions, mea.main_axes[1])]).T
    mea_pitch = [np.max(np.diff(np.sort(mea_pos[:, 0]))), np.max(np.diff(np.sort(mea_pos[:, 1])))]

    if spacing is None:
        spacing = 0.1 * np.min(mea_pitch)

    if mea_pitch[0] == 0:
        mea_pitch[0] = spacing + 1
    if mea_pitch[1] == 0:
        mea_pitch[1] = spacing + 1

    # normalize to min peak
    if vscale is None:
        signalmin = 1.5 * np.max(np.abs(signals))
        signals_norm = signals / signalmin * mea_pitch[1]
    else:
        signals_norm = signals / vscale * mea_pitch[1]

    if colors is None:
        if len(signals.shape) > 2:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        else:
            colors = 'k'

    number_electrodes = mea.number_electrodes
    if len(signals.shape) == 3:  # multiple
        for sp_i, sp in enumerate(signals_norm):
            for el in range(number_electrodes):
                if len(colors) > 1:
                    ax.plot(np.linspace(-(mea_pitch[0] - spacing) / 2., (mea_pitch[0] - spacing) / 2., signals.shape[2])
                            + mea_pos[el, 0], np.transpose(sp[el, :]) + mea_pos[el, 1], lw=lw,
                            color=colors[int(np.mod(sp_i, len(colors)))],
                            label='EAP ' + str(sp_i + 1))
                else:
                    ax.plot(np.linspace(-(mea_pitch[0] - spacing) / 2., (mea_pitch[0] - spacing) / 2., signals.shape[2])
                            + mea_pos[el, 0], np.transpose(sp[el, :]) + mea_pos[el, 1], lw=lw, color=colors,
                            label='EAP ' + str(sp_i + 1))

    else:
        for el in range(number_electrodes):
            ax.plot(np.linspace(-(mea_pitch[0] - spacing) / 2., (mea_pitch[0] - spacing) / 2., signals.shape[1]) +
                    mea_pos[el, 0], signals_norm[el, :] +
                    mea_pos[el, 1], color=colors, lw=lw)

    if hide_axis:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    ax.set_xlim(
        [np.min(mea_pos[:, 0]) - (mea_pitch[0] - spacing) / 2., np.max(mea_pos[:, 0]) + (mea_pitch[0] - spacing) / 2.])

    if scalebar:
        if time is None:
            raise AttributeError("Pass the 'time' argument")
        else:
            shift = 0.1 * spacing
            pos_h = [np.min(mea_pos[:, 0]), np.min(mea_pos[:, 1]) - 1.5 * mea_pitch[1]]
            if vscale is None:
                length_h = mea_pitch[1] * signalmin / (signalmin // 10 * 10)
            else:
                length_h = mea_pitch[1]
            pos_w = [np.min(mea_pos[:, 0]), np.min(mea_pos[:, 1]) - 1.5 * mea_pitch[1]]
            length_w = mea_pitch[0] / 5.

            ax.plot([pos_h[0], pos_h[0]], [pos_h[1], pos_h[1] + length_h], color='k', lw=2)
            if vscale is None:
                ax.text(pos_h[0] + shift, pos_h[1] + length_h / 2., str(int(signalmin // 10 * 10)) + ' $\mu$V')
            else:
                ax.text(pos_h[0] + shift, pos_h[1] + length_h / 2., str(int(vscale)) + ' $\mu$V')
            ax.plot([pos_w[0], pos_w[0] + length_w], [pos_w[1], pos_w[1]], color='k', lw=2)
            ax.text(pos_w[0] + shift, pos_w[1] - length_h / 3., str(time / 5) + ' ms')

    if not no_tight:
        fig.tight_layout()

    if axis_equal:
        ax.axis('equal')

    return ax


def play_mea_recording(signals, mea, fs, window=1, step=0.1, colors=None, lw=1, ax=None, spacing=None,
                       vscale=None, repeat=False, interval=10, hide_axis=True):
    '''
    Plays animation of the signals at electrode locations.

    Parameters
    ----------
    signals: np.array
        The signals to plot. Can be 2D (single signals) or 3D (multiple signals)
    mea: MEA object
        The MEA to be plotted
    fs: float
        Sampling frequency in Hz
    window: float
        The sliding window in seconds
    step: float
        The step of the sliding window in seconds
    colors: matplotlib colors
        The color or colors to use
    lw: float
        Line width of the lines
    ax: matplotlib axis
        The axis to plot on
    spacing: float
        The spacing in the x-direction
    vscale: float
        The scale to use for the signal (default .5 times the maximum signal)
    repeat: bool
        If True (default=False), animation is repeated
    interval: int
        Interval between frames in ms
    hide_axis: bool
        If True (default), axis are hidden

    Returns
    -------
    anim: matplotlib animation
        The output animation

    '''
    import matplotlib.animation as animation

    n_window = int(fs * window)
    n_step = int(fs * step)
    start = np.arange(0, signals.shape[1], n_step)

    mea_pos = np.array([np.dot(mea.positions, mea.main_axes[0]), np.dot(mea.positions, mea.main_axes[1])]).T
    mea_pitch = [np.max(np.diff(mea_pos[:, 0])), np.max(np.diff(mea_pos[:, 1]))]

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, frameon=False)
    else:
        fig = ax.get_figure()

    if spacing is None:
        spacing = 0.1 * np.min(mea_pitch)

    # normalize to min peak
    if vscale is None:
        signalmin = 1.5 * np.max(np.abs(signals))
        signals_norm = signals / signalmin * mea_pitch[1]
    else:
        signals_norm = signals / vscale * mea_pitch[1]

    if colors is None:
        if len(signals.shape) > 2:
            colors = plt.rcParams['axes.color_cycle']
        else:
            colors = 'k'

    number_electrodes = mea.number_electrodes
    lines = []
    for el in range(number_electrodes):
        if len(signals.shape) == 3:  # multiple
            raise Exception('Dimensions should be Nchannels x Nsamples')
        else:
            line, = ax.plot(
                np.linspace(-(mea_pitch[0] - spacing) / 2., (mea_pitch[0] - spacing) / 2. - spacing, n_window)
                + mea_pos[el, 0], np.zeros(n_window) + mea_pos[el, 1], color=colors, lw=lw)
            lines.append(line)

    text = ax.text(0.7, 0, 'Time: ',
                   color='k', fontsize=15, transform=ax.transAxes)

    if hide_axis:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    def update(i):
        if n_window + i < signals.shape[1]:
            for el in range(number_electrodes):
                lines[el].set_ydata(signals_norm[el, i:n_window + i] + mea_pos[el, 1])
        else:
            for el in range(number_electrodes):
                lines[el].set_ydata(np.pad(signals_norm[el, i:],
                                           (0, n_window - (signals.shape[1] - i)), 'constant') + mea_pos[el, 1])

        text.set_text('Time: ' + str(round(i / float(fs), 1)) + ' s')

        return tuple(lines) + (text,)

    anim = animation.FuncAnimation(fig, update, start, interval=interval, blit=True, repeat=repeat)
    fig.tight_layout()

    return anim


def plot_cylinder_3d(bottom, direction, length, radius, color='k', alpha=.5, ax=None,
                     xlim=None, ylim=None, zlim=None):
    '''
    Plots cylinder in 3d.

    Parameters
    ----------
    bottom: list or np.array
        3D point of the bottom of the cylinder
    direction: list or np.array
        3D direction of the cylinder
    length: float
        Length of the cylinder in um
    radius: float
        Radius of the cylinder in um
    color: matplotlib color
        Color of the cylinder
    alpha: float
        Alpha vlue of the cylinder
    ax: matplotlib axis
        The axis to plot on
    xlim: list
        Limits for x axis
    ylim: list
        Limits for y axis
    ylim: list
        Limits for z axis

    Returns
    -------
    ax: matplotlib axis
        The output axis
    '''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    poly3d = _get_polygons_for_cylinder(bottom, direction, length, radius, n_points=100, facecolor=color, edgecolor='k',
                                        alpha=alpha, lw=0., flatten_along_zaxis=False)

    for crt_poly3d in poly3d:
        ax.add_collection3d(crt_poly3d)

    if xlim:
        ax.set_xlim3d(xlim)
    if ylim:
        ax.set_xlim3d(ylim)
    if zlim:
        ax.set_xlim3d(zlim)

    return ax


def _rotate_poly3Dcollection(poly3d, axis_1, axis_2, shift=None):
    '''
    Helper function to rotate poly3Csollections in 3d.
    '''
    if shift is None:
        shift = [0, 0, 0]
    vec = poly3d._vec
    shift = np.array(shift)
    M = np.array([axis_1, axis_2, np.cross(axis_1, axis_2)])  # Get the rotation matrix

    rotated_vec = np.array([np.append(np.dot(M.T, v[:3]) + shift, v[-1]) for v in vec.T]).T
    poly3d._vec = rotated_vec


def _make_3d_ellipse_patch(size, axis_1, axis_2, position, ax, facecolor='orange', edgecolor=None, alpha=1.):
    '''
    Helper function to make 3d ellipse patch.
    '''
    p = patches.Circle((0, 0), size, facecolor=facecolor, alpha=alpha)
    ax.add_patch(p)

    path = p.get_path()  # Get the path and the associated transform
    trans = p.get_patch_transform()
    #
    path = trans.transform_path(path)  # Apply the transform

    p.__class__ = art3d.PathPatch3D  # Change the class
    p._code3d = path.codes  # Copy the codes
    p._facecolor3d = p.get_facecolor  # Get the face color

    verts = path.vertices  # Get the vertices in 2D

    M = np.array([axis_1, axis_2, np.cross(axis_1, axis_2)])  # Get the rotation matrix

    p._segment3d = np.array([np.dot(M.T, (x, y, 0)) for x, y in verts])
    p._segment3d += position

    return p


def _cylinder(pos_start, direction, length, radius, n_points, flatten_along_zaxis=False):
    '''
    Helper function to construct 3d cylinders.
    '''
    alpha = np.array([0., length])

    theta_ring = np.linspace(0., np.pi * 2., n_points)
    r = radius

    x = np.zeros((theta_ring.size * alpha.size))
    y = np.zeros((theta_ring.size * alpha.size))
    z = np.zeros((theta_ring.size * alpha.size))

    for idx_alpha, crt_alpha in enumerate(alpha):
        x[idx_alpha * theta_ring.size:
          (idx_alpha + 1) * theta_ring.size] = \
            r * np.cos(theta_ring)
        y[idx_alpha * theta_ring.size:
          (idx_alpha + 1) * theta_ring.size] = \
            r * np.sin(theta_ring)
        z[idx_alpha * theta_ring.size:
          (idx_alpha + 1) * theta_ring.size] = \
            crt_alpha * np.ones(theta_ring.size)

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    z = np.atleast_2d(z)

    d = direction

    # rot1, phi
    r_1 = np.array([0., 1., 0.])
    # rot2, theta
    r_2 = np.array([0., 0., 1.])

    # fix negative angles
    if d[0] == 0:
        theta = -np.sign(d[1]) * np.pi / 2.
    else:
        if d[0] > 0:
            theta = -np.arctan(d[1] / d[0])
        else:
            theta = np.pi - np.arctan(d[1] / d[0])

    rho = np.sqrt((d[0] ** 2 + d[1] ** 2))

    if rho == 0:
        phi = 0.
    else:
        phi = -(np.pi / 2. - np.arctan(d[2] / rho))

    # print 'phi: ', np.rad2deg(phi)

    rot1_m = rotation_matrix(r_1, phi)
    rot2_m = rotation_matrix(r_2, theta)

    for idx, (crt_x, crt_y, crt_z) in enumerate(zip(x[0], y[0], z[0])):
        crt_v = np.array([crt_x, crt_y, crt_z])
        crt_v = np.dot(crt_v, rot1_m)
        crt_v = np.dot(crt_v, rot2_m)
        x[0][idx] = crt_v[0]
        y[0][idx] = crt_v[1]
        z[0][idx] = crt_v[2]

    x += pos_start[0]
    y += pos_start[1]
    z += pos_start[2]
    if flatten_along_zaxis is True:
        z = np.abs(z)
        z *= 0.00000000001
    return x, y, z


def _get_polygons_for_cylinder(pos_start, direction, length, radius, n_points, facecolor='b', edgecolor='k', alpha=1.,
                               lw=0., flatten_along_zaxis=False):
    '''
    Helper function to construct polygons from cylinders.
    '''
    x, y, z = _cylinder(pos_start,
                        direction,
                        length,
                        radius,
                        n_points,
                        flatten_along_zaxis)

    alpha_tup = alpha,
    edge_col = mpl_colors.to_rgb(edgecolor) + alpha_tup
    face_col = mpl_colors.to_rgb(facecolor) + alpha_tup

    theta_ring = np.linspace(0., np.pi * 2., n_points)
    verts_hull = []
    for idx_theta, crt_theta in enumerate(theta_ring):
        if idx_theta <= theta_ring.size - 2:
            x_verts = [x[0][idx_theta],
                       x[0][idx_theta + 1],
                       x[0][idx_theta + 1 + theta_ring.size],
                       x[0][idx_theta + theta_ring.size]]
            y_verts = [y[0][idx_theta],
                       y[0][idx_theta + 1],
                       y[0][idx_theta + 1 + theta_ring.size],
                       y[0][idx_theta + theta_ring.size]]
            z_verts = [z[0][idx_theta],
                       z[0][idx_theta + 1],
                       z[0][idx_theta + 1 + theta_ring.size],
                       z[0][idx_theta + theta_ring.size]]
            verts_hull.append(list(zip(x_verts, y_verts, z_verts)))

    poly3d_hull = []
    for crt_vert in verts_hull:
        cyl = Poly3DCollection([crt_vert], linewidths=lw)
        cyl.set_facecolor(face_col)
        cyl.set_edgecolor(edge_col)

        poly3d_hull.append(cyl)

    # draw lower lid
    x_verts = x[0][0:theta_ring.size - 1]
    y_verts = y[0][0:theta_ring.size - 1]
    z_verts = z[0][0:theta_ring.size - 1]
    verts_lowerlid = list(zip(x_verts, y_verts, z_verts))
    poly3ed_lowerlid = Poly3DCollection([verts_lowerlid], linewidths=lw, zorder=1)
    poly3ed_lowerlid.set_facecolor(face_col)
    poly3ed_lowerlid.set_edgecolor(edge_col)

    # draw upper lid
    x_verts = x[0][theta_ring.size:theta_ring.size * 2 - 1]
    y_verts = y[0][theta_ring.size:theta_ring.size * 2 - 1]
    z_verts = z[0][theta_ring.size:theta_ring.size * 2 - 1]
    verts_upperlid = list(zip(x_verts, y_verts, z_verts))
    poly3ed_upperlid = Poly3DCollection([verts_upperlid], linewidths=lw, zorder=1)
    poly3ed_upperlid.set_facecolor(face_col)
    poly3ed_upperlid.set_edgecolor(edge_col)

    return_col = poly3d_hull
    return_col.append(poly3ed_lowerlid)
    return_col.append(poly3ed_upperlid)
    return return_col
