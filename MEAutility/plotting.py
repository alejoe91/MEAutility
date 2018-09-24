from __future__ import print_function

import numpy as np
from .core import MEA, RectMEA, rotation_matrix
import matplotlib
import pylab as plt


def plot_probe(mea, ax=None, xlim=None, ylim=None):
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
    import matplotlib.patches as patches
    from matplotlib.path import Path
    from matplotlib.collections import PatchCollection

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    n_elec = mea.positions.shape[0]
    elec_size = mea.size
    mea_pos = np.array([np.dot(mea.positions, mea.main_axes[0]), np.dot(mea.positions, mea.main_axes[1])]).T

    min_x, max_x = [np.min(np.dot(mea.positions, mea.main_axes[0])),
                    np.max(np.dot(mea.positions, mea.main_axes[0]))]
    center_x = (min_x + max_x)/2.
    min_y, max_y = [np.min(np.dot(mea.positions, mea.main_axes[1])),
                    np.max(np.dot(mea.positions, mea.main_axes[1]))]
    center_y = (min_y + max_y)/2.

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

    path = Path(verts, codes)

    patch = patches.PathPatch(path, facecolor='green', edgecolor='k', lw=0.5, alpha=0.3)
    ax.add_patch(patch)

    if mea.shape == 'square':
        for e in range(n_elec):
            elec = patches.Rectangle((mea_pos[e, 0] - elec_size, mea_pos[e, 1] - elec_size), 2*elec_size,  2*elec_size,
                                     alpha=0.7, facecolor='orange', edgecolor=[0.3, 0.3, 0.3], lw=0.5)

            ax.add_patch(elec)
    elif mea.shape == 'circle':
        for e in range(n_elec):
            elec = patches.Circle((mea_pos[e, 0], mea_pos[e, 1]), elec_size,
                                     alpha=0.7, facecolor='orange', edgecolor=[0.3, 0.3, 0.3], lw=0.5)

            ax.add_patch(elec)

    ax.set_xlim(probe_left - 5*elec_size, probe_right + 5*elec_size)
    ax.set_ylim(probe_bottom - 5*elec_size, probe_top + 5*elec_size)
    ax.axis('equal')

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    return ax


def plot_probe_3d(mea, alpha=.5, ax=None, xlim=None, ylim=None, zlim=None, top=1000, type='shank'):
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

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    elec_list = []

    if mea.shape == 'square':
        for pos in mea.positions:
                # elec = patches.Rectangle((mea_pos[e, 0] - elec_size, mea_pos[e, 1] - elec_size), 2 * elec_size,
                #                          2 * elec_size,
                #                          alpha=0.7, facecolor='orange', edgecolor=[0.3, 0.3, 0.3], lw=0.5)
            elec = np.array([pos - mea.size * mea.main_axes[0] - mea.size * mea.main_axes[1],
                             pos - mea.size * mea.main_axes[0] + mea.size * mea.main_axes[1],
                             pos + mea.size * mea.main_axes[0] + mea.size * mea.main_axes[1],
                             pos + mea.size * mea.main_axes[0] - mea.size * mea.main_axes[1]])

            elec_list.append(elec)

        el = Poly3DCollection(elec_list)
        alpha = (0.7,)
        el_col = mpl_colors.to_rgb('orange') + alpha
        el.set_facecolor(el_col)
        ax.add_collection3d(el)
    elif mea.shape == 'circle':
        for pos in mea.positions:
            p = make_3d_ellipse_patch(mea.size, mea.main_axes[0], mea.main_axes[1],
                                  pos, ax, facecolor='orange', edgecolor=None, alpha=0.7)

    min_x, max_x = [np.min(np.dot(mea.positions, mea.main_axes[0])) * mea.main_axes[0],
                    np.max(np.dot(mea.positions, mea.main_axes[0])) * mea.main_axes[0]]
    center_x = (min_x + max_x) / 2.
    min_y, max_y = [np.min(np.dot(mea.positions, mea.main_axes[1])) * mea.main_axes[1],
                    np.max(np.dot(mea.positions, mea.main_axes[1])) * mea.main_axes[1]]
    center_y = (min_y + max_y) / 2.

    if type == 'shank':
        probe_height = 200
        probe_top = max_y + probe_height * mea.main_axes[1]
        probe_bottom = min_y - probe_height * mea.main_axes[1]
        probe_corner = min_y - 0.1 * probe_height * mea.main_axes[1]
        probe_left = min_x - 0.1 * probe_height * mea.main_axes[0]
        probe_right = max_x + 0.1 * probe_height * mea.main_axes[0]

        verts = np.array([
            min_x - 2 * mea.size * mea.main_axes[0] + probe_top,  # left, bottom
            min_x - 2 * mea.size * mea.main_axes[0] + probe_corner,  # left, top
            center_x + probe_bottom,  # right, top
            max_x + 2 * mea.size * mea.main_axes[0] + probe_corner,  # right, bottom
            max_x + 2 * mea.size * mea.main_axes[0] + probe_top,
        ])
    elif type == 'planar':
        verts = np.array([
            min_x - 2 * mea.size * mea.main_axes[0] + max_y + 2 * mea.size * mea.main_axes[1],  # left, bottom
            min_x - 2 * mea.size * mea.main_axes[0] + (min_y - 2 * mea.size * mea.main_axes[1]),
            max_x + 2 * mea.size * mea.main_axes[0] + (min_y - 2 * mea.size * mea.main_axes[1]),
            max_x + 2 * mea.size * mea.main_axes[0] + max_y + 2 * mea.size * mea.main_axes[1],
        ])

    r = Poly3DCollection([verts])
    # r.set_facecolor('green')
    alpha = (0.3,)
    mea_col = mpl_colors.to_rgb('g') + alpha
    edge_col = mpl_colors.to_rgb('k') + alpha
    r.set_edgecolor(edge_col)
    r.set_facecolor(mea_col)
    ax.add_collection3d(r)


    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        xlim = [np.min(mea.positions[:,0]), np.max(mea.positions[:,0])]
        print(xlim)
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ylim = [np.min(mea.positions[:, 1]), np.max(mea.positions[:, 1])]
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)
    else:
        zlim = [np.min(mea.positions[:, 2]), np.max(mea.positions[:, 2])]
        ax.set_zlim(zlim)

    return ax

# # TODO 3d surf plot
# def plot_v_plane(mea, plane, offset=100):
#     '''
#
#     Parameters
#     ----------
#     mea
#     plane
#
#     Returns
#     -------
#
#     '''
#     if plane == 'xy':
#         x_vec = np.arange(1, bound, unit)
#         y_vec = np.arange(-bound, bound, unit)
#         z_vec = np.arange(-bound, bound, unit)
#     elif plane == 'yz':
#
#     elif plane == 'xz':
#
#
#     x, y, z = np.meshgrid(x_vec, y_vec, z_vec)
#
#     v_grid = np.zeros((len(y_vec), len(z_vec)))
#
#     # maintain matrix orientation (row - z, column - y, [0,0] - top left corner)
#     z_vec = z_vec[::-1]
#
#     for ii in range(len(z_vec)):
#         for jj in range(len(y_vec)):
#             v_grid[ii, jj] = mea.compute_field(np.array([15, y_vec[jj], z_vec[ii]]))
#
#
#     fig = plt.figure(figsize=[6, 16])
#     gs = gridspec.GridSpec(9,
#                            10,
#                            hspace=0.,
#                            wspace=0.)
#     fig.subplots_adjust(left=0.01, right=.8, top=1., bottom=0.01)
#     elev = 30
#     azim = -60
#     dist = 10
#     # Add surface
#     y_plane, z_plane = np.meshgrid(y_vec, z_vec)
#
#     v_grid_orig = np.zeros((len(y_vec), len(z_vec)))
#
#     # maintain matrix orientation (row - z, column - y, [0,0] - top left corner)
#
#     for ii in range(len(z_vec)):
#         for jj in range(len(y_vec)):
#             v_grid_orig[ii, jj] = mea.compute_field(np.array(
#                 [15, y_plane[ii][jj], z_plane[ii][jj]]))
#
#     # ax1 = fig.add_subplot(311, projection='3d')
#     ax1 = fig.add_subplot(gs[0:3, 0:9], projection='3d')
#     # ax1 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
#
#     ax1.view_init(elev=elev, azim=azim)
#     surf1 = ax1.plot_surface(y_plane,
#                              z_plane,
#                              v_grid_orig,
#                              cmap=cm.coolwarm,
#                              alpha=0.3,
#                              zorder=0,
#                              antialiased=True)
#     # ax1.contour(y_plane,
#     #             z_plane,
#     #             v_grid_orig,
#     #             cmap=cm.coolwarm,
#     #             extend3d=True,)
#
#     ax1.set_xticklabels([])
#     ax1.set_yticklabels([])
#     ax1.set_zticklabels([])
#     # Get rid of the panes
#     ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax1.w_zaxis.set_pane_color((1.0, 1.0, 1., 0.0))
#
#     # Get rid of the spines
#     ax1.w_xaxis.line.set_color((1.0, 1.0, 1., 0.0))
#     ax1.w_yaxis.line.set_color((1.0, 1.0, 1., 0.0))
#     ax1.w_zaxis.line.set_color((1.0, 1.0, 1., 0.0))
#     ax1.dist = 10..
#     cax1 = fig.add_subplot(gs[1, 9:])
#     cbar_ax1 = fig.colorbar(surf1, cax=cax1)
#     cbar_ax1.set_label('mV', rotation=270)
#
#     ax2 = fig.add_subplot(gs[3:6, 0:9], projection='3d')
#     cax2 = fig.add_subplot(gs[4, 9:])
#     ax2.view_init(elev=elev, azim=azim)
#     ax2.set_xlim3d(-30, 30)
#     ax2.set_ylim3d(-30, 30)
#     ax2.set_zlim3d(0, 30)
#     ax2.dist = 10..
#     soma_length = 3.
#     soma_radius = 1.
#     axon_length = 15.
#     axon_radius = .2
#     n_points = 20.
#
#     verts = []
#     elec_size = 5
#     for e in range(mea.number_electrode):
#         yy = [mea.electrodes[e].position[1] - elec_size,
#               mea.electrodes[e].position[1] - elec_size,
#               mea.electrodes[e].position[1] + elec_size,
#               mea.electrodes[e].position[1] + elec_size]
#         zz = [mea.electrodes[e].position[2] + elec_size,
#               mea.electrodes[e].position[2] - elec_size,
#               mea.electrodes[e].position[2] - elec_size,
#               mea.electrodes[e].position[2] + elec_size]
#         xx = [0, 0, 0, 0]
#         verts.append(list(zip(yy, zz, xx)))
#
#     jet = plt.get_cmap('jet')
#     colors = mea.get_currents() / np.max(np.abs(mea.get_current_matrix())) + 1
#     curr = ax2.add_collection3d(Poly3DCollection(verts,
#                                                  #                                            zorder=1,
#                                                  alpha=0.8,
#                                                  color=jet(colors)))
#     currents = mea.get_currents() / 1000
#
#     m = cm.ScalarMappable(cmap=cm.jet)
#     bounds = np.round(np.linspace(np.min(currents), np.max(currents), 7))
#     norm = mpl_colors.BoundaryNorm(bounds, cm.jet)
#     m.set_array(currents)
#     cbar_ax2 = plt.colorbar(m, cax=cax2, norm=norm, boundaries=bounds)
#     cbar_ax2.set_label('mA', rotation=270)
#     # ghost_axis = ax2.scatter(xx, yy, zz, color=jet(colors))
#     # cax2 = fig.add_subplot(gs[4, 9:])
#     # fig.colorbar(ghost_axis, cax=cax2)
#     # ghost_axis.axis('off')
#     # cmap = plt.cm.jet
#     # norm = mpl.colors.BoundaryNorm(colors, cmap)
#
#     # cb = mpl.colorbar.ColorbarBase(ax2,
#     #                                cmap=cax2,
#     #                                norm=norm)
#
#     ax2.set_xticklabels([])
#     ax2.set_yticklabels([])
#
#     # Get rid of the panes
#     ax2.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax2.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax2.w_zaxis.set_pane_color((1.0, 1.0, 1., 0.0))
#
#     # Get rid of the spines
#     ax2.w_xaxis.line.set_color((1.0, 1.0, 1., 0.0))
#     ax2.w_yaxis.line.set_color((1.0, 1.0, 1., 0.0))
#     ax2.w_zaxis.line.set_color((1.0, 1.0, 1., 0.0))
#
#     # ax2.set_xlabel('Y [um]')
#     # ax2.set_ylabel('Z [um]')
#     ax2.set_zlabel('Z [um]')
#
#     # last axis
#     ax0 = fig.add_subplot(gs[6:9, 0:9], projection='3d')
#
#     ax0.view_init(elev=elev, azim=azim)
#     ax0.dist = 10..
#     # plot data points.
#     # ax0.scatter([mea.electrodes[elec].position[1] for elec in range(0, mea.number_electrode)],
#     #             [mea.electrodes[elec].position[2] for elec in range(0, mea.number_electrode)],
#     #             marker='o', c='b', s=50, zorder=2)
#     ax0.set_xlabel('y ($\mu$m)', fontsize=20)
#     ax0.set_ylabel('z ($\mu$m)', fontsize=20)
#     ax0.xaxis.set_tick_params(labelsize=15, width=5)
#     ax0.yaxis.set_tick_params(labelsize=15, width=5)
#
#     # ax0.set_xticklabels([])
#     # ax0.set_yticklabels([])
#     ax0.set_zticks([])
#     # Get rid of the panes
#     ax0.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax0.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#     ax0.w_zaxis.set_pane_color((1.0, 1.0, 1., 0.0))
#
#     # Get rid of the spines
#     ax0.w_xaxis.line.set_color((1.0, 1.0, 1., 0.0))
#     ax0.w_yaxis.line.set_color((1.0, 1.0, 1., 0.0))
#     ax0.w_zaxis.line.set_color((1.0, 1.0, 1., 0.0))
#
#     ax0.grid(b=False)
#
#     ax0.set_zlim3d(0, 0.1)
#
#     CS = ax0.contourf(y_vec,
#                       z_vec,
#                       v_grid_orig,
#                       zdir='z',
#                       offset=-0.0001,
#                       cmap=cm.coolwarm)


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

    verts_3d = np.array([(x, y, 0) for x, y in verts])
    verts_3d_rot = np.array([np.dot(M, (x, y, 0)) for x, y in verts])

    print(verts_3d)
    print(verts_3d_rot)
    print(M)

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
