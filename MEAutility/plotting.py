from __future__ import print_function

import numpy as np
from .core import MEA, RectMEA, rotation_matrix
import matplotlib
import pylab as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
import MEAutility as MEA
from matplotlib.patches import Ellipse
import matplotlib.animation as animation
from matplotlib.collections import PolyCollection
from matplotlib import colors as mpl_colors
import mpl_toolkits.mplot3d as a3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_probe(mea_pos, mea_pitch, shape='square', elec_dim=10, axis=None, xlim=None, ylim=None):
    '''

    Parameters
    ----------
    mea_pos
    mea_pitch
    shape
    elec_dim
    axis
    xlim
    ylim

    Returns
    -------

    '''
    from matplotlib.path import Path
    import matplotlib.patches as patches
    from matplotlib.collections import PatchCollection

    if axis:
        ax = axis
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    n_elec = mea_pos.shape[0]

    y_pitch = mea_pitch[0]
    z_pitch = mea_pitch[1]


    elec_size = elec_dim / 2
    elec_size = (np.min([y_pitch,z_pitch]) - 0.3*np.min([y_pitch,z_pitch]))/2.
    elec_dim = (np.min([y_pitch,z_pitch]) - 0.3*np.min([y_pitch,z_pitch]))

    min_y = np.min(mea_pos[:,1])
    max_y = np.max(mea_pos[:,1])
    min_z = np.min(mea_pos[:,2])
    max_z = np.max(mea_pos[:,2])
    center_y = 0
    probe_height = 200
    probe_top = max_z + probe_height
    prob_bottom = min_z - probe_height
    prob_corner = min_z - 0.1*probe_height
    probe_left = min_y - 0.1*probe_height
    probe_right = max_y + 0.1*probe_height



    verts = [
        (min_y - 2*elec_dim, probe_top),  # left, bottom
        (min_y - 2*elec_dim, prob_corner),  # left, top
        (center_y, prob_bottom),  # right, top
        (max_y + 2*elec_dim, prob_corner),  # right, bottom
        (max_y + 2*elec_dim, probe_top),
        (min_y - 2 * elec_dim, max_z + 2 * elec_dim) # ignored
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

    if shape == 'square':
        for e in range(n_elec):
            elec = patches.Rectangle((mea_pos[e, 1] - elec_size, mea_pos[e, 2] - elec_size), elec_dim,  elec_dim,
                                     alpha=0.7, facecolor='orange', edgecolor=[0.3, 0.3, 0.3], lw=0.5)

            ax.add_patch(elec)
    elif shape == 'circle':
        for e in range(n_elec):
            elec = patches.Circle((mea_pos[e, 1], mea_pos[e, 2]), elec_size,
                                     alpha=0.7, facecolor='orange', edgecolor=[0.3, 0.3, 0.3], lw=0.5)

            ax.add_patch(elec)

    ax.set_xlim(probe_left - 5*elec_dim, probe_right + 5*elec_dim)
    ax.set_ylim(prob_bottom - 5*elec_dim, probe_top + 5*elec_dim)
    # ax.axis('equal')

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)


def plot_probe_3d(mea_pos, rot_axis, theta, pos=[0, 0, 0], shape='square', alpha=.5,
                  elec_dim=15, probe_name=None, ax=None, xlim=None, ylim=None, zlim=None, top=1000):
    '''

    Parameters
    ----------
    mea_pos
    mea_pitch
    shape
    elec_dim
    axis
    xlim
    ylim

    Returns
    -------

    '''
    from matplotlib.patches import Circle

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    M = rotation_matrix(rot_axis, theta)
    rot_pos = np.dot(M, mea_pos.T).T
    rot_pos += np.array(pos)

    normal = np.cross(rot_pos[1]-rot_pos[0], rot_pos[-1]-rot_pos[0])

    if probe_name is not None:
        if 'neuronexus' in probe_name.lower():
            for elec in rot_pos:
                p = Circle((0, 0), elec_dim/2., facecolor='orange', alpha=alpha)
                ax.add_patch(p)
                _make_patch_3d(p, rot_axis, theta+np.pi/2.)
                _pathpatch_translate(p, elec)

        tip_el_y = np.min(mea_pos[:, 2])
        bottom = tip_el_y - 62
        cz = 62 + np.sqrt(22**2 - 18**2) + 9*25
        top = top

        x_shank = [0, 0, 0, 0, 0, 0, 0]
        y_shank = [-57, -57, -31, 0, 31, 57, 57]
        z_shank = [bottom + top, bottom + cz, bottom + 62, bottom, bottom + 62, bottom + cz, bottom + top]

        shank_coord = np.array([x_shank, y_shank, z_shank])
        shank_coord_rot = np.dot(M, shank_coord)

        r = Poly3DCollection([np.transpose(shank_coord_rot)])
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

    return rot_pos


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



def _make_patch_3d(pathpatch, rot_axis, angle, z=0):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color

    verts = path.vertices #Get the vertices in 2D

    M = rotation_matrix(rot_axis, angle) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])

def _pathpatch_2d_to_3d(pathpatch, z = 0, normal = 'z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0,0,0), index)

    normal /= np.linalg.norm(normal) #Make sure the vector is normalised

    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color

    verts = path.vertices #Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1)) #Obtain the rotation vector
    M = rotation_matrix(d) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])


def _pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta

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
