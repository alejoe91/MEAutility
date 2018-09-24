from __future__ import print_function

import numpy as np
from .core import MEA, RectMEA, rotation_matrix
import matplotlib
import pylab as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.collections import PatchCollection


from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
# import MEAutility as MEA
# from matplotlib.patches import Ellipse
# import matplotlib.animation as animation
# from matplotlib.collections import PolyCollection
from matplotlib import colors as mpl_colors
# import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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
    ax.set_ylim(prob_bottom - 5*elec_size, probe_top + 5*elec_size)
    ax.axis('equal')

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    return ax


def plot_probe_3d(mea, alpha=.5, ax=None, xlim=None, ylim=None, zlim=None, top=1000):
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

    Returns
    -------

    '''
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

    probe_height = 200
    probe_top = max_y + probe_height * mea.main_axes[1]
    probe_bottom = min_y - probe_height * mea.main_axes[1]
    probe_corner = min_y - 0.1 * probe_height * mea.main_axes[1]
    probe_left = min_x - 0.1 * probe_height * mea.main_axes[0]
    probe_right = max_x + 0.1 * probe_height * mea.main_axes[0]


    verts = np.array([
        min_x - 2 * mea.size * mea.main_axes[0] + probe_top,  # left, bottom
        min_x - 2 * mea.size * mea.main_axes[0] + probe_corner,  # left, top
        center_x - probe_bottom,  # right, top
        max_x + 2 * mea.size * mea.main_axes[0] + probe_corner,  # right, bottom
        max_x + 2 * mea.size * mea.main_axes[0] + probe_top,
    ])

    raise Exception()

    r = Poly3DCollection([verts])
    # r.set_facecolor('green')
    alpha = (0.3,)
    mea_col = mpl_colors.to_rgb('g') + alpha
    edge_col = mpl_colors.to_rgb('k') + alpha
    r.set_edgecolor(edge_col)
    r.set_facecolor(mea_col)
    ax.add_collection3d(r)


    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if zlim:
        ax.set_zlim(zlim)

    # return rot_pos


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

    M = [axis_1, axis_2, np.cross(axis_1, axis_2)]  # Get the rotation matrix

    p._segment3d = np.array([np.dot(M, (x, y, 0)) for x, y in verts])
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
