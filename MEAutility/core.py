from __future__ import print_function
"""

Collection of classes and functions for MEA stimulation

Conventions:
position = [um]
current = [nA]
voltage = [mV]

"""

import numpy as np
from numpy import linalg as la
import os.path
import copy
import yaml
import shutil
from distutils.version import StrictVersion

if StrictVersion(yaml.__version__) >= StrictVersion('5.0.0'):
    use_loader = True
else:
    use_loader = False

class Electrode:
    def __init__(self, position=None, normal=None, current=None, sigma=None, max_field=None, shape=None, size=None):
        '''

        Parameters
        ----------
        position
        current
        sigma
        max_field
        points
        shape
        size
        '''
        # convert to numpy array
        if type(position) == list:
            position = np.array(position)
        self.position = position
        if sigma is not None:
            self.sigma = sigma
        else:
            self.sigma = 0.3
        if max_field is not None:
            self.max_field = max_field
        else:
            self.max_field = 1000
        if current is not None:
            self.current = current
        else:
            self.current = 0
        if normal is None and self.points > 1:
            raise Exception('Provide normal direction to electrode')
        elif normal is not None:
            self.normal = normal
        else:
            self.normal = None
        if shape is not None:
            self.shape = shape
        else:
            self.shape = 'square'
        if size is not None:
            self.size = size
        else:
            self.size = 5


    def field_contribution(self, pos, npoints=1, model='inf', main_axes=None, seed=None):
        '''

        Parameters
        ----------
        pos
        npoints
        model
        main_axes

        Returns
        -------

        '''
        if seed is not None:
            np.random.seed(seed)
        potential, stim_points = [], []
        if isinstance(self.current, (float, int, np.float, np.integer)):
            if self.current != 0:
                if npoints == 1:
                    stim_points = self.position
                    if any(pos != self.position):
                        if model == 'inf':
                            potential = self.current / (4 * np.pi * self.sigma * la.norm(pos - self.position))
                        elif model == 'semi':
                            potential = self.current / (2 * np.pi * self.sigma * la.norm(pos - self.position))
                    else:
                        print("WARNING: point and electrode location are coincident! Field set to MAX_FIELD: ",
                              self.max_field)
                        potentiel = self.max_field
                else:
                    stim_points = []
                    spl = 0
                    for p in np.arange(npoints):
                        placed = False
                        if self.shape == 'square':
                            if main_axes is None:
                                raise Exception("If electrode is 'square' main_axes should be provided")
                            while not placed:
                                arr = (2 * self.size) * np.random.rand(3) - self.size
                                # rotate to align to main_axes and keep uniform distribution
                                M = np.array([main_axes[0], main_axes[1], self.normal])
                                arr_rot = np.dot(M.T, arr)
                                point = np.cross(arr_rot, self.normal)  # + self.position
                                if np.abs(np.dot(point, main_axes[0])) < self.size and \
                                        np.abs(np.dot(point, main_axes[1])) < self.size:
                                    placed = True
                                    stim_points.append(point + self.position)
                        else:
                            while not placed:
                                arr = (2 * self.size) * np.random.rand(3) - self.size
                                M = np.array([main_axes[0], main_axes[1], self.normal])
                                arr_rot = np.dot(M.T, arr)
                                point = np.cross(arr_rot, self.normal)
                                if np.linalg.norm(point) < self.size:
                                    placed = True
                                    stim_points.append(point + self.position)

                    stim_points = np.array(stim_points)
                    split_current = float(self.current) / npoints
                    potential = 0
                    for el_pos in stim_points:
                        spl += split_current
                        if any(pos != el_pos):
                            if model == 'inf':
                                potential += split_current / (4 * np.pi * self.sigma * la.norm(pos - el_pos))
                            elif model == 'semi':
                                potential += split_current / (2 * np.pi * self.sigma * la.norm(pos - el_pos))
                        else:
                            print("WARNING: point and electrode location are coincident! Field set to MAX_FIELD: ",
                                  self.max_field)
                            potential += self.max_field
            else:
                potential = 0
        elif isinstance(self.current, (list, np.ndarray)):
            potential = np.zeros(len(self.current))
            for i, c in enumerate(self.current):
                if c != 0:
                    if npoints == 1:
                        stim_points = self.position
                        if any(pos != self.position):
                            for i, c in enumerate(self.current):
                                if model == 'inf':
                                    potential[i] = c / (4 * np.pi * self.sigma * la.norm(pos - self.position))
                                elif model == 'semi':
                                    potential[i] = c / (2 * np.pi * self.sigma * la.norm(pos - self.position))
                        else:
                            print("WARNING: point and electrode location are coincident! Field set to MAX_FIELD: ",
                                  self.max_field)
                            potential = np.array([self.max_field] * len(self.current))
                    else:
                        stim_points = []
                        for p in np.arange(npoints):
                            placed = False
                            if self.shape == 'square':
                                if main_axes is None:
                                    raise Exception("If electrode is 'square' main_axes should be provided")
                                while not placed:
                                    arr = (2 * self.size) * np.random.rand(3) - self.size
                                    # rotate to align to main_axes and keep uniform distribution
                                    M = np.array([main_axes[0], main_axes[1], self.normal])
                                    arr_rot = np.dot(M.T, arr)
                                    point = np.cross(arr_rot, self.normal) # + self.position
                                    if np.abs(np.dot(point, main_axes[0])) < self.size and \
                                            np.abs(np.dot(point, main_axes[1])) < self.size:
                                        placed=True
                                        stim_points.append(point + self.position)
                            else:
                                while not placed:
                                    arr = (2 * self.size) * np.random.rand(3) - self.size
                                    point = np.cross(arr, self.normal) + self.position
                                    if np.linalg.norm(point - self.position) < self.size:
                                        placed=True
                                        stim_points.append(point)

                        stim_points = np.array(stim_points)
                        split_current = c / npoints
                        for el_pos in stim_points:
                            if any(pos != el_pos):
                                if model == 'inf':
                                    potential[i] += split_current / (4 * np.pi * self.sigma * la.norm(pos - el_pos))
                                elif model == 'semi':
                                    potential[i] += split_current / (2 * np.pi * self.sigma * la.norm(pos - el_pos))
                            else:
                                print("WARNING: point and electrode location are coincident! Field set to MAX_FIELD: ",
                                      self.max_field)
                                potential[i] += self.max_field
                else:
                    potential[i] = 0
                    stim_points = np.zeros((npoints, 3))
        return potential, stim_points


class MEA(object):
    '''This class handles properties and stimulation of general multi-electrode arrays
    '''
    def __init__(self, positions, info, normal=None, points_per_electrode=None, sigma=None):
        '''

        Parameters
        ----------
        positions
        info
        model
        sigma
        '''
        self.number_electrodes = len(positions)
        if sigma == None:
            self.sigma = 0.3
        else:
            self.sigma = float(sigma)

        if points_per_electrode == None:
            self.points_per_electrode = 1
        else:
            self.points_per_electrode = int(points_per_electrode)

        if 'shape' in info.keys():
            self.shape = info['shape']
        else:
            self.shape = 'square'

        if 'plane' in info.keys():
            self.plane = info['plane']
        else:
            self.plane = 'yz'

        if 'size' in info.keys():
            self.size = info['size']
        else:
            self.size = 5

        if 'type' in info.keys():
            self.type = info['type']
        else:
            self.type = 'mea'

        if 'model' in info.keys():
            self.model = info['model']
        else:
            self.model = 'semi'

        # Assumption (electrodes on the same plane)
        if self.number_electrodes > 1:
            if normal is None:
                self.normal = np.cross(positions[0], positions[1])
                self.normal /= np.linalg.norm(self.normal)
            else:
                if isinstance(normal, (list, np.ndarray)):
                    self.normal = normal
                else:
                    self.normal = np.cross(positions[0], positions[1])
                    self.normal /= np.linalg.norm(self.normal)
        else:
            self.normal = np.cross(self.main_axes[0], self.main_axes[1])

        # print("Model is set to %s" % self.model)

        if self.plane == 'xy':
            self.main_axes = np.array([[1, 0, 0], [0, 1, 0]])
        elif self.plane == 'yz':
            self.main_axes = np.array([[0, 1, 0], [0, 0, 1]])
        elif self.plane == 'xz':
            self.main_axes = np.array([[1, 0, 0], [0, 0, 1]])

        self.electrodes = [Electrode(pos, normal=self.normal, sigma=self.sigma, shape=self.shape,
                                     size=self.size) for pos in positions]
        self.number_electrodes = len(self.electrodes)

        self._positions = None
        self.info = info


    @property
    def positions(self):
        return self._get_electrode_positions()


    @property
    def currents(self):
        return self._get_currents()


    @currents.setter
    def currents(self, current_values):
        if not isinstance(current_values, (list, np.ndarray)) or \
                len(current_values) != self.number_electrodes:
            raise Exception("Number of currents should be equal to number of electrodes %d" % self.number_electrodes)
        for i, el in enumerate(self.electrodes):
            el.current = current_values[i]

    # override [] method
    def __getitem__(self, index):
        # return row of current matrix
        if index < self.number_electrodes:
            return self.electrodes[index]
        else:
            print("Index out of bound")
            return None

    def _set_positions(self, positions):
        '''

        Parameters
        ----------
        electrodes

        Returns
        -------

        '''
        for i, el in enumerate(self.electrodes):
            el.position = positions[i]

    def _set_normal(self, normal):
        '''

        Parameters
        ----------
        electrodes

        Returns
        -------

        '''
        for i, el in enumerate(self.electrodes):
            el.normal = normal/np.linalg.norm(normal)

    def _get_currents(self):
        '''

        Returns
        -------

        '''
        curr = [el.current for el in self.electrodes]
        if np.all([isinstance(c, (list, np.ndarray)) for c in curr]):
            curr_len = [len(el.current) for el in self.electrodes]
            if len(np.unique(curr_len)) == 1:
                tcurrent = np.unique(curr_len)[0]
                currents = np.zeros((self.number_electrodes, tcurrent))
                for i, el in enumerate(self.electrodes):
                    currents[i] = el.current
            else:
                self.reset_currents()
                print("Currents have different lengths! Currents has been resetted.")
        elif np.any([isinstance(c, (list, np.ndarray)) for c in curr]):
            # this deals with setting an array current accessing the electrode directly
            # find length if current array
            for i, el in enumerate(self.electrodes):
                if isinstance(el.current, (list, np.ndarray)):
                    tcurrent = len(el.current)
                    break
                else:
                    pass
            currents = np.zeros((self.number_electrodes, tcurrent))
            for i, el in enumerate(self.electrodes):
                if isinstance(el.current, (list, np.ndarray)) and len(el.current) == tcurrent:
                    currents[i] = el.current
                elif isinstance(el.current, (list, np.ndarray)) and len(el.current) != tcurrent:
                    currents[i] = [el.current[0]] * tcurrent
                else:
                    currents[i] = [el.current] * tcurrent
            self.currents = currents
        else:
            currents = np.zeros(self.number_electrodes)
            for i, el in enumerate(self.electrodes):
                currents[i] = el.current
        return currents

    def _get_electrode_positions(self):
        pos = np.zeros((self.number_electrodes, 3))
        for i, el in enumerate(self.electrodes):
            pos[i, :] = el.position
        return pos

    def set_electrodes(self, electrodes):
        '''

        Parameters
        ----------
        electrodes

        Returns
        -------

        '''
        self.electrodes = electrodes
        self.number_electrodes = len(electrodes)

    def set_random_currents(self, mean=0, sd=1000):
        '''

        Parameters
        ----------
        amp

        Returns
        -------

        '''
        currents = sd * np.random.randn(self.number_electrodes) + mean
        self.currents = currents

    def set_currents(self, current_values):
        '''

        Parameters
        ----------
        current_values

        Returns
        -------

        '''
        if isinstance(current_values, (list, np.ndarray)):
            if len(current_values) != self.number_electrodes:
                raise Exception("Number of currents should be equal to number of electrodes %d" % self.number_electrodes)
            else:
                for i, el in enumerate(self.electrodes):
                    el.current = current_values[i]
        else:
            raise Exception("Current values should be a list or np.array with len=%d" % self.number_electrodes)

    def set_current(self, el_id, current_value):
        '''

        Parameters
        ----------
        current_values

        Returns
        -------

        '''
        if isinstance(current_value, (float, int, np.float, np.integer)):
            self.electrodes[el_id].current = current_value
        elif isinstance(current_value, (list, np.ndarray)):
            for el_i, el in enumerate(self.electrodes):
                if el_i == el_id:
                    el.current = np.array(current_value)
                elif isinstance(el.current, (float, int, np.float, np.integer)):
                        el.current = np.array([el.current] * len(current_value))
                elif isinstance(el.current, (list, np.ndarray)):
                    if len(el.current) != len(current_value):
                        el.current = np.array([el.current[0]] * len(current_value))

    def reset_currents(self, amp=None):
        '''

        Parameters
        ----------
        amp

        Returns
        -------

        '''
        if amp is None:
            currents = np.zeros(self.number_electrodes)
        else:
            currents = amp*np.ones(self.number_electrodes)
        self.currents = currents

    def compute_field(self, points, return_stim_points=False, seed=None, verbose=False):
        '''

        Parameters
        ----------
        points
        return_stim_points
        seed

        Returns
        -------

        '''
        c = self.electrodes[0].current

        if isinstance(points, list):
            points = np.array(points)

        if points.ndim == 1:
            if len(points) != 3:
                print("Error: expected 3d point")
                return
            else:
                if isinstance(c, (float, int, np.float, np.integer)):
                    vp = 0
                    stim_points = []
                    for ii in np.arange(self.number_electrodes):
                        vs, sp = self.electrodes[ii].field_contribution(points, npoints=self.points_per_electrode,
                                                                        model=self.model, main_axes=self.main_axes,
                                                                        seed=seed)

                        vp += vs
                        if len(sp) != 0:
                            stim_points.append(sp)
                elif isinstance(c, (list, np.ndarray)):
                    vp = np.zeros(len(c))
                    stim_points = []
                    for ii in np.arange(self.number_electrodes):
                        vs, sp = self.electrodes[ii].field_contribution(points, npoints=self.points_per_electrode,
                                                                        model=self.model, main_axes=self.main_axes,
                                                                        seed=seed)
                        vp += vs
                        if len(sp) != 0:
                            stim_points.append(sp)
        elif points.ndim == 2:
            if points.shape[1] != 3:
                print("Error: expected 3d points")
                return
            else:
                if isinstance(c, (float, int, np.float, np.integer)):
                    vp = np.zeros(points.shape[0])
                    for pp in np.arange(len(vp)):
                        if verbose:
                            print("Computing point: ", pp+1)
                        pf = 0
                        stim_points = []
                        cur_point = points[pp]
                        for ii in np.arange(self.number_electrodes):
                            vs, sp = self.electrodes[ii].field_contribution(cur_point, npoints=self.points_per_electrode,
                                                                         model=self.model, main_axes=self.main_axes)
                            pf += vs
                            if len(sp) != 0:
                                stim_points.append(sp)
                        vp[pp] = pf
                elif isinstance(c, (list, np.ndarray)):
                    vp = np.zeros((points.shape[0], len(c)))
                    stim_points = []
                    for pp in np.arange(len(vp)):
                        if verbose:
                            print("Computing point: ", pp+1)
                        pf = np.zeros(len(c))
                        stim_points = []
                        cur_point = points[pp]
                        for ii in np.arange(self.number_electrodes):
                            vs, sp = self.electrodes[ii].field_contribution(cur_point, npoints=self.points_per_electrode,
                                                                            model=self.model, main_axes=self.main_axes,
                                                                            seed=seed)
                            pf += vs
                            if len(sp) != 0:
                                stim_points.append(sp)
                        vp[pp] = pf
        stim_points = np.array(stim_points)
        if len(stim_points.shape) == 3:
            stim_points = np.reshape(stim_points, (stim_points.shape[0]*stim_points.shape[1], stim_points.shape[2]))
        if return_stim_points:
            return vp, stim_points
        else:
            return vp

    def save_currents(self, filename):
        np.save(filename, self.currents)
        print('Currents saved successfully to file ', filename)

    def load_currents(self, filename):
        if os.path.isfile(filename):
            currents = np.load(filename)
            if len(currents) != self.number_electrodes:
                print('Error: number of currents in file different than number of electrodes')
            else:
                print('Currents loaded successfully from file ', filename)
                self.currents = currents
        else:
            print('File does not exist')

    def rotate(self, axis, theta):
        '''

        Parameters
        ----------
        axis: np.array
            rotation axis
        theta: float
            anglo in degrees counterclock wise

        Returns
        -------

        '''
        M = rotation_matrix(axis, np.deg2rad(theta))
        rot_pos = np.dot(M, self.positions.T).T
        rot_pos = np.round(rot_pos, 3)
        rot_axis = np.dot(M, self.main_axes.T).T
        self.main_axes = np.round(rot_axis, 3)
        normal = np.cross(rot_pos[1] - rot_pos[0], rot_pos[-1] - rot_pos[0])
        self.normal = np.round(normal, 3)
        self.normal /= np.linalg.norm(self.normal)

        self._set_positions(rot_pos)
        self._set_normal(normal)

    def move(self, vector):
        '''

        Parameters
        ----------
        axis
        theta

        Returns
        -------

        '''
        move_pos = self.positions + vector
        self._set_positions(move_pos)

    def center(self):
        '''

        Returns
        -------

        '''
        center_pos = center_mea(self.positions)
        self._set_positions(center_pos)


class RectMEA(MEA):
    '''

    '''
    def __init__(self, positions, info):
        '''

        Parameters
        ----------
        dim
        pitch
        width
        x_plane
        '''
        MEA.__init__(self, positions, info)
        if 'dim' in info.keys():
            self.dim = info['dim']
        else:
            raise AttributeError("Rectangular MEA should have 'dim' field in info")
        if 'pitch' in info.keys():
            self.pitch = info['pitch']
        else:
            raise AttributeError("Rectangular MEA should have 'pitch' field in info")
        if isinstance(self.dim, (int, np.integer)):
            self.dim = [self.dim, self.dim]
        if isinstance(self.pitch, (int, float)):
            self.pitch = [self.pitch, self.pitch]

    # override [] method
    def __getitem__(self, index):
        # return row of current matrix
        if index < self.dim[0]:
            electrode_matrix = self.get_electrode_matrix()
            return electrode_matrix[index]
        else:
            print("Index out of bound")
            return None

    def get_current_matrix(self):
        current_matrix = np.zeros(self.dim)
        for i in np.arange(0, self.dim[0]):
            for j in np.arange(0, self.dim[1]):
                current_matrix[i, j] = self.currents[self.dim[0] * j + i]
        return current_matrix

    def get_electrode_matrix(self):
        electrode_matrix = np.empty(self.dim, dtype=object)
        for i in np.arange(0, self.dim[0]):
            for j in np.arange(0, self.dim[1]):
                electrode_matrix[i, j] = self.electrodes[self.dim[0] * j + i]
        return electrode_matrix

    def set_current_matrix(self, currents):
        current_array = np.zeros((self.number_electrodes))
        for i in np.arange(self.dim[0]):
            for j in np.arange(self.dim[1]):
                current_array[self.dim[0] * i + j] = currents[j, i]
        self.set_currents(current_values=current_array)

    def get_linear_id(self, i, j):
        return self.dim[0] * j + i


def add_3dim(pos2d, plane, offset=None):
    '''

    Parameters
    ----------
    plane

    Returns
    -------

    '''
    nelec = pos2d.shape[0]
    if plane == 'xy':
        pos = np.hstack((pos2d, offset * np.ones((nelec, 1))))
    elif plane == 'yz':
        pos = np.hstack((offset * np.ones((nelec, 1)), pos2d))
    elif plane == 'xz':
        pos = np.hstack((pos2d, offset * np.ones((nelec, 1))))
        pos[:, [0, 1, 2]] = pos[:, [0, 2, 1]]
    return pos


def center_mea(pos):
    '''

    Parameters
    ----------
    pos

    Returns
    -------

    '''
    return pos - np.mean(pos, axis=0, keepdims=True)


def get_positions(elinfo, center=True):
    '''Computes the positions of the elctrodes based on the elinfo

    Parameters
    ----------
    elinfo: dict
        Contains electrode information from yaml file (dim, pitch, sortlist, plane, pos)

    Returns
    -------
    positions: np.array
        3d points with the centers of the electrodes

    '''
    electrode_pos = False
    # method 1: positions in elinfo
    if 'pos' in elinfo.keys():
        pos = np.array(elinfo['pos'])
        nelec = pos.shape[0]
        if len(pos.shape) == 1:
            if len(pos) == 2:
                pos2d = np.array([pos])
                if 'plane' not in elinfo.keys():
                    # print("'plane' field with 2D dimensions assumed to be 'yz")
                    plane = 'yz'
                else:
                    plane = elinfo['plane']
                if 'offset' not in elinfo.keys():
                    offset = 0
                else:
                    offset = elinfo['offset']
                pos = add_3dim(pos2d, plane, offset)
            elif len(pos) == 3:
                pos = np.array([pos])
            elif len(pos) != 3:
                raise AttributeError('pos attribute should be one or a list of 2D or 3D points')
        elif len(pos.shape) == 2:
            if pos.shape[1] == 2:
                pos2d = pos
                if 'plane' not in elinfo.keys():
                    # print("'plane' field with 2D dimensions assumed to be 'yz")
                    plane = 'yz'
                else:
                    plane = elinfo['plane']
                if 'offset' not in elinfo.keys():
                    offset = 0
                else:
                    offset = elinfo['offset']
                pos = add_3dim(pos2d, plane, offset)
            elif pos.shape[1] != 3:
                raise AttributeError('pos attribute should be a list of 2D or 3D points')
        electrode_pos = True

    # method 2: dim, pithch, stagger
    if 'dim' in elinfo.keys():
        dim = elinfo['dim']
        if dim == 1:
            if 'plane' not in elinfo.keys():
                # print("'plane' field with 2D dimensions assumed to be 'yz")
                plane = 'yz'
            else:
                plane = elinfo['plane']
            if 'offset' not in elinfo.keys():
                offset = 0
            else:
                offset = elinfo['offset']
            pos2d = np.array([[0, 0]])
            pos = add_3dim(pos2d, plane, offset)
        else:
            if 'pitch' not in elinfo.keys():
                raise AttributeError("When 'dim' is used, also 'pitch' should be specified.")
            else:
                pitch = elinfo['pitch']

            if isinstance(dim, (int, np.integer)):
                dim = [dim, dim]
            if isinstance(pitch, (int, np.integer)) or isinstance(pitch, (float, np.float)):
                pitch = [pitch, pitch]
            if len(dim) == 2:
                d1 = np.array([])
                d2 = np.array([])
                if 'stagger' in elinfo.keys():
                    stagger = elinfo['stagger']
                else:
                    stagger = None
                for d_i in np.arange(dim[1]):
                    if stagger is not None:
                        if isinstance(stagger, (int, np.integer)) or isinstance(stagger, (float, np.float)):
                            if np.mod(d_i, 2):
                                d1new = np.arange(dim[0]) * pitch[0] + stagger
                            else:
                                d1new = np.arange(dim[0]) * pitch[0]
                        elif len(stagger) == len(dim):
                            d1new = np.arange(dim[0]) * pitch[0] + stagger[d_i]
                        else:
                            d1new = np.arange(dim[0]) * pitch[0]
                    else:
                        d1new = np.arange(dim[0]) * pitch[0]
                    d1 = np.concatenate((d1, d1new))
                    d2 = np.concatenate((d2, dim[0] * [pitch[1] * d_i]))
                pos2d = np.vstack((d2, d1)).T
                if 'plane' not in elinfo.keys():
                    # print("'plane' field with 2D dimensions assumed to be 'yz")
                    plane = 'yz'
                else:
                    plane = elinfo['plane']
                if 'offset' not in elinfo.keys():
                    offset = 0
                else:
                    offset = elinfo['offset']
                pos2d = np.concatenate((np.reshape(d2.T, (d1.size, 1)),
                                        np.reshape(d1.T, (d2.size, 1))), axis=1)
                pos = add_3dim(pos2d, plane, offset)

            elif len(dim) >= 3:
                d1 = np.array([])
                d2 = np.array([])
                if 'stagger' in elinfo.keys():
                    stagger = elinfo['stagger']
                else:
                    stagger = None
                for d_i, d in enumerate(dim):
                    if stagger is not None:
                        if isinstance(stagger, (int, np.integer)) or isinstance(stagger, (float, np.float)):
                            if np.mod(d_i, 2):
                                d1new = np.arange(d) * pitch[0] + stagger
                            else:
                                d1new = np.arange(d) * pitch[0]
                        elif len(stagger) == len(dim):
                            d1new = np.arange(d) * pitch[0] + stagger[d_i]
                        else:
                            d1new = np.arange(d) * pitch[0]
                    else:
                        d1new = np.arange(d) * pitch[0]
                    d1 = np.concatenate((d1, d1new))
                    d2 = np.concatenate((d2, d * [pitch[1] * d_i]))
                pos2d = np.vstack((d2, d1)).T
                if 'plane' not in elinfo.keys():
                    # print("'plane' field with 2D dimensions assumed to be 'yz")
                    plane = 'yz'
                else:
                    plane = elinfo['plane']
                if 'offset' not in elinfo.keys():
                    offset = 0
                else:
                    offset = elinfo['offset']
                pos = add_3dim(pos2d, plane, offset)
        electrode_pos = True

    if electrode_pos and center:
        centered_pos = center_mea(pos)
        # resort electrodes in case
        centered_pos_sorted = copy.deepcopy(centered_pos)
        if 'sortlist' in elinfo.keys() and elinfo['sortlist'] is not None:
            sortlist = elinfo['sortlist']
            for i, si in enumerate(sortlist):
                centered_pos_sorted[si] = centered_pos[i]
        else:
            centered_pos_sorted = centered_pos
        return centered_pos_sorted
    elif electrode_pos and not center:
        # resort electrodes in case
        pos_sorted = copy.deepcopy(pos)
        if 'sortlist' in elinfo.keys() and elinfo['sortlist'] is not None:
            sortlist = elinfo['sortlist']
            for i, si in enumerate(sortlist):
                pos_sorted[si] = pos[i]
        else:
            pos_sorted = pos
        return pos_sorted
    else:
        print("Define either a list of positions 'pos' or 'dim' and 'pitch'")
        return None


def check_if_rect(elinfo):
    if 'dim' in elinfo.keys():
        dim = elinfo['dim']
        if isinstance(dim, (int, np.integer)):
            return True
        elif isinstance(dim, list):
            if len(dim) <= 2:
                return True
        return False


def return_mea(electrode_name=None, info=None):
    '''

    Parameters
    ----------
    electrode_name

    Returns
    -------

    '''
    if electrode_name is None and info is None:
        this_dir, this_filename = os.path.split(__file__)
        electrodes = [f[:-5] for f in os.listdir(os.path.join(this_dir, "electrodes"))]
        print('Available MEA: \n', electrodes)
        return
    elif electrode_name is not None:
        # load MEA info
        this_dir, this_filename = os.path.split(__file__)
        electrode_path = os.path.join(this_dir, "electrodes")
        if os.path.isfile(os.path.join(electrode_path, electrode_name + '.yaml')):
            with open(os.path.join(electrode_path, electrode_name + '.yaml')) as meafile:
                if use_loader:
                    elinfo = yaml.load(meafile, Loader=yaml.FullLoader)
                else:
                    elinfo = yaml.load(meafile)
            pos = get_positions(elinfo)
            # create MEA object
            if check_if_rect(elinfo):
                mea = RectMEA(positions=pos, info=elinfo)
            else:
                mea = MEA(positions=pos, info=elinfo)
            return mea
        elif os.path.isfile(os.path.join(electrode_path, electrode_name + '.yml')):
            with open(os.path.join(electrode_path, electrode_name + '.yml')) as meafile:
                if use_loader:
                    elinfo = yaml.load(meafile, Loader=yaml.FullLoader)
                else:
                    elinfo = yaml.load(meafile)
            pos = get_positions(elinfo)
            # create MEA object
            if check_if_rect(elinfo):
                mea = RectMEA(positions=pos, info=elinfo)
            else:
                mea = MEA(positions=pos, info=elinfo)
            return mea
        else:
            print("MEA model named %s not found" % electrode_name)
            this_dir, this_filename = os.path.split(__file__)
            electrodes = [f[:-5] for f in os.listdir(os.path.join(this_dir, "electrodes"))]
            print('Available MEA: \n', electrodes)
            return
    elif info is not None:
        elinfo = info
        if 'center' in elinfo.keys():
            center = elinfo['center']
        else:
            center=True
        pos = get_positions(elinfo, center=center)
        # create MEA object
        if check_if_rect(elinfo):
            mea = RectMEA(positions=pos, info=elinfo)
        else:
            mea = MEA(positions=pos, info=elinfo)
        return mea


def return_mea_info(electrode_name=None):
    '''

    Parameters
    ----------
    electrode_name

    Returns
    -------

    '''
    if electrode_name is None:
        this_dir, this_filename = os.path.split(__file__)
        electrodes = [f[:-5] for f in os.listdir(os.path.join(this_dir, "electrodes"))]
        print('Available MEA: \n', electrodes)
        return
    else:
        # load MEA info
        this_dir, this_filename = os.path.split(__file__)
        electrode_path = os.path.join(this_dir, "electrodes")
        if os.path.isfile(os.path.join(electrode_path, electrode_name + '.yaml')):
            with open(os.path.join(electrode_path, electrode_name + '.yaml')) as meafile:
                if use_loader:
                    elinfo = yaml.load(meafile, Loader=yaml.FullLoader)
                else:
                    elinfo = yaml.load(meafile)
            return elinfo
        elif os.path.isfile(os.path.join(electrode_path, electrode_name + '.yml')):
            with open(os.path.join(electrode_path, electrode_name + '.yml')) as meafile:
                if use_loader:
                    elinfo = yaml.load(meafile, Loader=yaml.FullLoader)
                else:
                    elinfo = yaml.load(meafile)
            return elinfo
        else:
            print("MEA model named %s not found" % electrode_name)
            this_dir, this_filename = os.path.split(__file__)
            electrodes = [f[:-5] for f in os.listdir(os.path.join(this_dir, "electrodes"))]
            print('Available MEA: \n', electrodes)
            return


def return_mea_list():
    '''

    Returns
    -------

    '''
    this_dir, this_filename = os.path.split(__file__)
    electrodes = [f[:-5] for f in os.listdir(os.path.join(this_dir, "electrodes"))]
    return electrodes


def add_mea(mea_yaml_path):
    '''Adds the mea design defined by the yaml file in the install folder

    Parameters
    ----------
    mea_yaml_file

    Returns
    -------

    '''
    path = os.path.abspath(mea_yaml_path)

    if path.endswith('.yaml') or path.endswith('.yml') and os.path.isfile(path):
        with open(path, 'r') as meafile:
            if use_loader:
                elinfo = yaml.load(meafile, Loader=yaml.FullLoader)
            else:
                elinfo = yaml.load(meafile)
            if 'pos' not in elinfo.keys():
                if 'dim' in elinfo.keys():
                    if elinfo['dim'] != 1 and 'pitch' not in elinfo.keys():
                        raise AttributeError("The yaml file should contin either a list of 3d or 2d positions 'pos' or "
                                             "intormation about dimension and pitch ('dim' and 'pitch')")
                else:
                    raise AttributeError("The yaml file should contin either a list of 3d or 2d positions 'pos' or "
                                         "intormation about dimension and pitch ('dim' and 'pitch') - unless dim=1")

        this_dir, this_filename = os.path.split(__file__)
        shutil.copy(path, os.path.join(this_dir, 'electrodes'))
        if path.endswith('.yaml'):
            electrodes = [f[:-5] for f in os.listdir(os.path.join(this_dir, "electrodes"))]
        elif path.endswith('.yml'):
            electrodes = [f[:-4] for f in os.listdir(os.path.join(this_dir, "electrodes"))]
        print('Available MEA: \n', electrodes)
        return


def remove_mea(mea_name):
    '''Adds the mea design defined by the yaml file in the install folder

    Parameters
    ----------
    mea_yaml_file

    Returns
    -------

    '''
    this_dir, this_filename = os.path.split(__file__)
    electrodes = [f for f in os.listdir(os.path.join(this_dir, "electrodes"))]
    for e in electrodes:
        if mea_name in e:
            if os.path.isfile(os.path.join(this_dir, "electrodes", mea_name + '.yaml')):
                os.remove(os.path.join(this_dir, "electrodes", mea_name + '.yaml'))
                print("Removed: ", os.path.join(this_dir, "electrodes", mea_name + '.yaml'))
            elif os.path.isfile(os.path.join(this_dir, "electrodes", mea_name + '.yml')):
                os.remove(os.path.join(this_dir, "electrodes", mea_name + '.yml'))
                print("Removed: ", os.path.join(this_dir, "electrodes", mea_name + '.yml'))
    electrodes = [f[:-5] for f in os.listdir(os.path.join(this_dir, "electrodes"))]
    print('Available MEA: \n', electrodes)
    return


def rotation_matrix(axis, theta):
    '''

    Parameters
    ----------
    axis
    theta

    Returns
    -------

    '''
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

#
# if __name__ == '__main__':
#     # test
#     # elinfo = {'pos': [[10,25],[10,-5],[10,5],[10,-25]]}
#     import matplotlib.pylab as plt
#     elinfo = {'dim': [10, 3], 'pitch': [10, 30]}
#     pos = get_positions(elinfo)
#
#     mea = return_mea(info=elinfo)
#     gpos = mea.positions
#     print(pos)
#
#     plt.plot(pos[:,1], pos[:,2], '*')
#     plt.plot(gpos[:,1], gpos[:,2], '*')
#     plt.axis('equal')