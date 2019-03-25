from __future__ import print_function

import numpy as np
from .core import MEA, RectMEA, rotation_matrix
import matplotlib
import matplotlib.pylab as plt

import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
from matplotlib import colors as mpl_colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_probe(mea, ax=None, xlim=None, ylim=None, color_currents=False, cmap='viridis', type='shank'):
    '''
    
    Parameters
    ----------
    mea
    axis
    xlim
    ylim

    Returns
    -------

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
        center_x = (min_x + max_x)/2.
        min_y, max_y = [np.min(np.dot(mea.positions, mea.main_axes[1])),
                        np.max(np.dot(mea.positions, mea.main_axes[1]))]
        center_y = (min_y + max_y)/2.

        if type == 'shank':
            probe_height = 200
            probe_top = max_y + probe_height
            probe_bottom = min_y - probe_height
            probe_corner = min_y - 0.1*probe_height
            probe_left = min_x - 0.1*probe_height
            probe_right = max_x + 0.1*probe_height

            verts = [
                (min_x - 2*elec_size, probe_top),  # left, bottom
                (min_x - 2*elec_size, probe_corner),  # left, top
                (center_x, probe_bottom),  # right, top
                (max_x + 2*elec_size, probe_corner),  # right, bottom
                (max_x + 2*elec_size, probe_top),
                (min_x - 2 * elec_size, max_y + 2 * elec_size) # ignored
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
                (max_x + 2 * elec_size, max_y + 2 * elec_size), # ignored
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

        patch = patches.PathPatch(path, facecolor='green', edgecolor='k', lw=0.5, alpha=0.3)
        ax.add_patch(patch)

        if color_currents:
            norm_curr = mea.currents / np.max(np.abs(mea.currents))
            colormap = plt.get_cmap(cmap)
            elec_colors = colormap(norm_curr)
        else:
            elec_colors = ['orange'] * mea.number_electrodes

        if mea.shape == 'square':
            for e in range(n_elec):
                elec = patches.Rectangle((mea_pos[e, 0] - elec_size, mea_pos[e, 1] - elec_size), 2*elec_size,  2*elec_size,
                                         alpha=0.7, facecolor=elec_colors[e], edgecolor=[0.3, 0.3, 0.3], lw=0.5)

                ax.add_patch(elec)
        elif mea.shape == 'circle':
            for e in range(n_elec):
                elec = patches.Circle((mea_pos[e, 0], mea_pos[e, 1]), elec_size,
                                         alpha=0.7, facecolor=elec_colors[e], edgecolor=[0.3, 0.3, 0.3], lw=0.5)

                ax.add_patch(elec)
    else:
        raise NotImplementedError('Wire type plotting not implemented')

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


def plot_probe_3d(mea, alpha=.5, ax=None, xlim=None, ylim=None, zlim=None, top=1000, type='shank',
                  color_currents=False, cmap='viridis'):
    '''

    Parameters
    ----------
    mea
    alpha
    ax
    xlim
    ylim
    zlim
    top
    type

    Returns
    -------

    '''
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d import art3d
    from matplotlib import colors as mpl_colors
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib import patches

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
            ax.add_collection3d(el)
        elif mea.shape == 'circle':
            for i, pos in enumerate(mea.positions):
                p = make_3d_ellipse_patch(mea.size, mea.main_axes[0], mea.main_axes[1],
                                      pos, ax, facecolor=elec_colors[i], edgecolor=None, alpha=0.7)

        center_probe = mea.positions.mean(axis=0)
        min_dim_1 = np.min(np.dot(mea.positions, mea.main_axes[0]))
        min_dim_2 = np.min(np.dot(mea.positions, mea.main_axes[1]))
        max_dim_1 = np.max(np.dot(mea.positions, mea.main_axes[0]))
        max_dim_2 = np.max(np.dot(mea.positions, mea.main_axes[1]))

        center_dim_1 = np.dot(center_probe, mea.main_axes[0])
        center_dim_2 = np.dot(center_probe, mea.main_axes[1])

        min_dim_1c = np.min(np.dot(mea.positions-center_probe, mea.main_axes[0]))
        min_dim_2c = np.min(np.dot(mea.positions-center_probe, mea.main_axes[1]))
        max_dim_1c = np.max(np.dot(mea.positions-center_probe, mea.main_axes[0]))
        max_dim_2c = np.max(np.dot(mea.positions-center_probe, mea.main_axes[1]))
        minmin, maxmax = [2*np.min([min_dim_1c, min_dim_2c]),
                          2*np.max([max_dim_1c, max_dim_2c])]

        if type == 'shank':
            probe_height = 200
            probe_top = (max_dim_2 + probe_height) * mea.main_axes[1]
            probe_bottom = (min_dim_2 - probe_height) * mea.main_axes[1]
            probe_corner = (min_dim_2 - 0.1 * probe_height) * mea.main_axes[1]
            probe_center_bottom = center_dim_1 * mea.main_axes[0]


            verts = np.array([
                (min_dim_1 - 2 * mea.size) * mea.main_axes[0] + probe_top,  # left, bottom
                (min_dim_1 - 2 * mea.size) * mea.main_axes[0] + probe_corner,  # left, top
                center_dim_1 * mea.main_axes[0] + probe_bottom,  # right, top
                (max_dim_1 + 2 * mea.size) * mea.main_axes[0] + probe_corner,  # right, bottom
                (max_dim_1 + 2 * mea.size) * mea.main_axes[0] + probe_top,
            ])

        elif type == 'planar':
            verts = np.array([
                (min_dim_1 - 3 * mea.size) * mea.main_axes[0] + (max_dim_2 + 3 * mea.size) * mea.main_axes[1],  # left, bottom
                (min_dim_1 - 3 * mea.size) * mea.main_axes[0] + (min_dim_2 - 3 * mea.size) * mea.main_axes[1],
                (max_dim_1 + 3 * mea.size) * mea.main_axes[0] + (min_dim_2 - 3 * mea.size) * mea.main_axes[1],
                (max_dim_1 + 3 * mea.size) * mea.main_axes[0] + (max_dim_2 + 3 * mea.size) * mea.main_axes[1],
            ])

        # verts += center_probe
        r = Poly3DCollection([verts])
        alpha = (0.3,)
        mea_col = mpl_colors.to_rgb('g') + alpha
        edge_col = mpl_colors.to_rgb('k') + alpha
        r.set_edgecolor(edge_col)
        r.set_facecolor(mea_col)
        ax.add_collection3d(r)


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
    else:
        raise NotImplementedError('Wire type plotting not implemented')

    return ax


def plot_v_image(mea, v_plane=None, x_bound=None, y_bound=None, z_bound=None, offset=0,
                 plane='yz', npoints=30, ax=None, cmap='viridis', **kwargs):
    '''

    Parameters
    ----------
    mea
    plane

    Returns
    -------

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
                plane='yz',  plot_plane=None, npoints=30, ax=None, cmap='viridis', alpha=0.8, distance=30, **kwargs):
    '''

    Parameters
    ----------
    mea
    plane

    Returns
    -------

    '''
    from mpl_toolkits.mplot3d import Axes3D

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
        # elif plane == 'par':
        #     assert x_bound is not None and z_bound is not None
        #     vec1 = np.linspace(x_bound[0], x_bound[1], npoints)
        #     vec2 = np.linspace(z_bound[0], z_bound[1], npoints)
        #     for i, v1 in enumerate(vec1):
        #         for j, v2 in enumerate(vec2):
        #             v_grid[i, j] = mea.compute_field(np.array([v1, offset, v2]))

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
                rotate_poly3Dcollection(surf1, [1, 0, 0], [0, 1, 0], shift=[0, 0, distance])
            elif plot_plane == 'yz':
                rotate_poly3Dcollection(surf1, [0, 1, 0], [0, 0, 1], shift=[distance, 0, 0])
            elif plot_plane == 'xz':
                rotate_poly3Dcollection(surf1, [1, 0, 0], [0, 0, 1], shift=[0, distance, 0])
            # elif plot_plane == 'par':


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
                rotate_poly3Dcollection(surf1, [1, 0, 0], [0, 1, 0], shift=[0, 0, distance])
            elif plot_plane == 'yz':
                rotate_poly3Dcollection(surf1, [0, 1, 0], [0, 0, 1], shift=[distance, 0, 0])
            elif plot_plane == 'xz':
                rotate_poly3Dcollection(surf1, [1, 0, 0], [0, 0, 1], shift=[0, distance, 0])

    return ax, v_grid.T


def plot_mea_recording(signals, mea, colors=None, points=False, lw=1, ax=None, spacing=None,
                       scalebar=False, time=None, dt=None, vscale=None):
    '''

    Parameters
    ----------
    signals
    mea_pos
    mea_pitch
    color
    points
    lw

    Returns
    -------

    '''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, frameon=False)
        no_tight = False
    else:
        no_tight = True

    mea_pos = np.array([np.dot(mea.positions, mea.main_axes[0]), np.dot(mea.positions, mea.main_axes[1])]).T
    mea_pitch = [np.max(np.diff(np.sort(mea_pos[:,0]))), np.max(np.diff(np.sort(mea_pos[:,1])))]

    if spacing is None:
        spacing = 0.1*np.max(mea_pitch)

    # normalize to min peak
    if vscale is None:
        signalmin = 1.5*np.max(np.abs(signals))
        signals_norm = signals / signalmin * mea_pitch[1]
    else:
        signals_norm = signals / vscale * mea_pitch[1]

    if colors is None:
        if len(signals.shape) > 2:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        else:
            colors = 'k'

    number_electrodes = mea.number_electrodes
    for el in range(number_electrodes):
        if len(signals.shape) == 3:  # multiple
            if points:
                for sp_i, sp in enumerate(signals_norm):
                    if len(colors) == len(signals_norm) and len(colors) > 1:
                        ax.plot(np.linspace(0, mea_pitch[0]-spacing, signals.shape[2]) + mea_pos[el, 0],
                                np.transpose(sp[el, :]) +  mea_pos[el, 1], linestyle='-', marker='o', ms=2, lw=lw,
                                color=colors[np.mod(sp_i, len(colors))],
                                label='EAP '+ str(sp_i+1))
                    else:
                        ax.plot(np.linspace(0, mea_pitch[0] - spacing, signals.shape[2]) + mea_pos[el, 0],
                                np.transpose(sp[el, :]) +
                                mea_pos[el, 1], linestyle='-', marker='o', ms=2, lw=lw, color=colors,
                                label='EAP ' + str(sp_i + 1))
            else:
                for sp_i, sp in enumerate(signals_norm):
                    if len(colors) == len(signals_norm):
                        ax.plot(np.linspace(0, mea_pitch[0]-spacing, signals.shape[2]) + mea_pos[el, 0],
                                np.transpose(sp[el, :]) + mea_pos[el, 1], lw=lw, color=colors[np.mod(sp_i, len(colors))],
                                label='EAP '+str(sp_i+1))
                    else:
                        ax.plot(np.linspace(0, mea_pitch[0] - spacing, signals.shape[2]) + mea_pos[el, 0],
                                np.transpose(sp[el, :]) + mea_pos[el, 1], lw=lw, color=colors,
                                label='EAP ' + str(sp_i + 1))

        else:
            if points:
                ax.plot(np.linspace(0, mea_pitch[0]-spacing, signals.shape[1]) + mea_pos[el, 0], signals_norm[el, :]
                        + mea_pos[el, 1], color=colors, linestyle='-', marker='o', ms=2, lw=lw)
            else:
                ax.plot(np.linspace(0, mea_pitch[0]-spacing, signals.shape[1]) + mea_pos[el, 0], signals_norm[el, :] +
                        mea_pos[el, 1], color=colors, lw=lw)

        # ax.set_ylim([np.min(signals), np.max(signals)])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    if scalebar:
        if dt is None and time is None:
            raise AttributeError('Pass either dt or time in the argument')
        else:
            shift = 0.1*spacing
            pos_h = [np.min(mea_pos[:, 0]), np.min(mea_pos[:, 1]) - 1.5*mea_pitch[1]]
            if vscale is None:
                length_h = mea_pitch[1] * signalmin / (signalmin // 10 * 10)
            else:
                length_h = mea_pitch[1]
            pos_w = [np.min(mea_pos[:, 0]), np.min(mea_pos[:, 1]) - 1.5*mea_pitch[1]]
            length_w = mea_pitch[0]/5.

            ax.plot([pos_h[0], pos_h[0]], [pos_h[1], pos_h[1] + length_h], color='k', lw=2)
            if vscale is None:
                ax.text(pos_h[0]+shift, pos_h[1] + length_h / 2., str(int(signalmin // 10 * 10)) + ' $\mu$V')
            else:
                ax.text(pos_h[0]+shift, pos_h[1] + length_h / 2., str(int(vscale)) + ' $\mu$V')
            ax.plot([pos_w[0], pos_w[0]+length_w], [pos_w[1], pos_w[1]], color='k', lw=2)
            ax.text(pos_w[0]+shift, pos_w[1]-length_h/3., str(time/5) + ' ms')

    if not no_tight:
        fig.tight_layout()

    return ax


def play_mea_recording(signals, mea, fs, window=1, step=0.1, colors=None, lw=1, ax=None, fig=None, spacing=None,
                       scalebar=False, time=None, dt=None, vscale=None, spikes=None, repeat=False, interval=10):
    '''

    Parameters
    ----------
    signals
    mea
    fs
    window
    step
    colors
    lw
    ax
    fig
    spacing
    scalebar
    time
    dt
    vscale
    spikes
    repeat
    interval

    Returns
    -------

    '''
    import matplotlib.animation as animation

    n_window = int(fs * window)
    n_step = int(fs * step)
    start = np.arange(0, signals.shape[1], n_step)

    mea_pos = np.array([np.dot(mea.positions, mea.main_axes[0]), np.dot(mea.positions, mea.main_axes[1])]).T
    mea_pitch = [np.max(np.diff(mea_pos[:, 0])), np.max(np.diff(mea_pos[:, 1]))]

    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(1, 1, 1, frameon=False)
        no_tight = False
    else:
        no_tight = True

    if spacing is None:
        spacing = 0.1 * np.max(mea_pitch)

    # normalize to min peak
    if vscale is None:
        signalmin = 1.5*np.max(np.abs(signals))
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
            line, = ax.plot(np.linspace(0, mea_pitch[0] - spacing, n_window) + mea_pos[el, 0],
                            np.zeros(n_window) + mea_pos[el, 1], color=colors, lw=lw)
            lines.append(line)

    text = ax.text(0.7, 0, 'Time: ',
                   color='k', fontsize=15, transform=ax.transAxes)

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

    anim = animation.FuncAnimation(fig, update, start, interval=interval, blit=True, repeat=False)
    fig.tight_layout()

    return anim



def plot_cylinder_3d(bottom, direction, length, radius, color='k', alpha=.5, ax=None,
                     xlim=None, ylim=None, zlim=None):
    '''

    Parameters
    ----------
    bottom
    direction
    color
    alpha
    ax
    xlim
    ylim
    zlim

    Returns
    -------

    '''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    poly3d = get_polygons_for_cylinder(bottom, direction, length, radius, n_points=100, facecolor=color, edgecolor='k',
                              alpha=alpha, lw=0., flatten_along_zaxis=False)

    for crt_poly3d in poly3d:
        ax.add_collection3d(crt_poly3d)

    if xlim:
        ax.set_xlim3d(xlim)
    if ylim:
        ax.set_xlim3d(ylim)
    if zlim:
        ax .set_xlim3d(zlim)

    return ax


def rotate_poly3Dcollection(poly3d, axis_1, axis_2, shift=[0,0,0]):
    vec = poly3d._vec
    shift = np.array(shift)
    M = np.array([axis_1, axis_2, np.cross(axis_1, axis_2)])  # Get the rotation matrix

    rotated_vec = np.array([np.append(np.dot(M.T, v[:3]) + shift, v[-1]) for v in vec.T]).T
    poly3d._vec = rotated_vec



def make_3d_ellipse_patch(size, axis_1, axis_2, position, ax, facecolor='orange', edgecolor=None, alpha=1):
    '''

    Parameters
    ----------
    size
    axis_1
    axis_2
    position
    ax
    facecolor
    edgecolor
    alpha

    Returns
    -------

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

    Parameters
    ----------
    pos_start
    direction
    length
    radius
    n_points
    flatten_along_zaxis

    Returns
    -------

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
        theta = -np.sign(d[1])*np.pi / 2.
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


def get_polygons_for_cylinder(pos_start, direction, length, radius, n_points, facecolor='b', edgecolor='k', alpha=1.,
                              lw = 0., flatten_along_zaxis=False):
    '''

    Parameters
    ----------
    pos_start
    direction
    length
    radius
    n_points
    facecolor
    edgecolor
    alpha
    lw
    flatten_along_zaxis

    Returns
    -------

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
            verts_hull.append([zip(x_verts, y_verts, z_verts)])

    poly3d_hull = []
    for crt_vert in verts_hull:
        cyl = Poly3DCollection(crt_vert, linewidths=lw)
        cyl.set_facecolor(face_col)
        cyl.set_edgecolor(edge_col)

        poly3d_hull.append(cyl)

    # draw lower lid
    x_verts = x[0][0:theta_ring.size - 1]
    y_verts = y[0][0:theta_ring.size - 1]
    z_verts = z[0][0:theta_ring.size - 1]
    verts_lowerlid = [zip(x_verts, y_verts, z_verts)]
    poly3ed_lowerlid = Poly3DCollection(verts_lowerlid, linewidths=lw, zorder=1)
    poly3ed_lowerlid.set_facecolor(face_col)
    poly3ed_lowerlid.set_edgecolor(edge_col)

    # draw upper lid
    x_verts = x[0][theta_ring.size:theta_ring.size * 2 - 1]
    y_verts = y[0][theta_ring.size:theta_ring.size * 2 - 1]
    z_verts = z[0][theta_ring.size:theta_ring.size * 2 - 1]
    verts_upperlid = [zip(x_verts, y_verts, z_verts)]
    poly3ed_upperlid = Poly3DCollection(verts_upperlid, linewidths=lw, zorder=1)
    poly3ed_upperlid.set_facecolor(face_col)
    poly3ed_upperlid.set_edgecolor(edge_col)

    return_col = poly3d_hull
    return_col.append(poly3ed_lowerlid)
    return_col.append(poly3ed_upperlid)
    return return_col
