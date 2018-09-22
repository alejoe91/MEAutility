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

class Electrode:
    #TODO: make grid contribution for squared electrodes
    def __init__(self, position=None, normal=None, current=None, sigma=None, max_field=None,
                 points=None, shape=None, size=None):
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
        if points is not None:
            self.points = points
        else:
            self.points = 1
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


    def set_current(self, current):
        self.current = current

    def set_position(self, pos):
        self.position = pos

    def set_normal(self, norm):
        self.normal = norm

    def set_sigma(self, sigma):
        self.sigma = sigma

    def set_max_field(self, max_field):
        self.max_field = max_field

    def field_contribution(self, pos, model='inf'):
        if self.points == 1:
            if any(pos != self.position):
                if model == 'inf':
                    return self.current / (4*np.pi*self.sigma*la.norm(pos-self.position))
                elif model == 'semi':
                    return self.current / (2*np.pi*self.sigma*la.norm(pos-self.position))
            else:
                print("WARNING: point and electrode location are coincident! Field set to MAX_FIELD: ", self.max_field)
                return self.max_field
        else:
            split_current = self.current / self.points
            potential = 0
            for p in range(self.points):
                elec_points = []
                if self.shape == 'square':
                    # draw random sample in 3D
                    arr = (2*self.size)*np.random.rand(3) - self.size
                    point = np.cross(arr, self.normal)
                    pass
                else:
                    pass
        # add return of x-y-z components of the field


class MEA(object):
    '''This class handles properties and stimulation of general multi-electrode arrays
    '''
    def __init__(self, positions, info, normal=None, model=None, sigma=None):
        '''

        Parameters
        ----------
        positions
        info
        model
        sigma
        '''
        # TODO add rotation axis and angle
        if sigma == None:
            self.sigma = 0.3
        else:
            self.sigma = float(sigma)

        # Assumption (electrodes on the same plane)
        if normal is None:
            self.normal = np.cross(positions[0], positions[1])
            self.normal /= np.linalg.norm(self.normal)
        else:
            pass

        self.electrodes = [Electrode(pos, normal=self.normal, sigma=self.sigma) for pos in positions]
        self.number_electrode = len(self.electrodes)
        if model is not None:
            if model == 'inf' or model=='semi':
                self.model = model
            else:
                raise AttributeError("Unknown model. Model can be 'inf' or 'semi'")
        else:
            self.model = 'inf'
            print("Setting model to 'inf'")

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
                len(current_values) != self.number_electrode:
            raise Exception("Number of currents should be equal to number of electrodes %d" % self.number_electrode)
        for i, el in enumerate(self.electrodes):
            el.set_current(current_values[i])


    def _set_positions(self, positions):
        '''

        Parameters
        ----------
        electrodes

        Returns
        -------

        '''
        for i, el in enumerate(self.electrodes):
            el.set_position(positions[i])


    def _set_normal(self, normal):
        '''

        Parameters
        ----------
        electrodes

        Returns
        -------

        '''
        for i, el in enumerate(self.electrodes):
            el.set_normal(normal/np.linalg.norm(normal))


    def _get_currents(self):
        '''

        Returns
        -------

        '''
        currents = np.zeros(self.number_electrode)
        for i, el in enumerate(self.electrodes):
            currents[i] = el.current
        return currents


    def _get_electrode_positions(self):
        pos = np.zeros((self.number_electrode, 3))
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
        self.number_electrode = len(electrodes)


    def set_random_currents(self, amp=None):
        '''

        Parameters
        ----------
        amp

        Returns
        -------

        '''
        if amp:
            currents = np.random.randn(self.number_electrode) * amp
        else:
            currents = np.random.randn(self.number_electrode) * 10
        self.currents(currents)

    def reset_currents(self, amp=None):
        '''

        Parameters
        ----------
        amp

        Returns
        -------

        '''
        if amp is None:
            currents = np.zeros(self.number_electrode)
        else:
            currents = amp*np.ones(self.number_electrode)
        self.currents(currents)


    def compute_field(self, points):
        '''

        Parameters
        ----------
        points

        Returns
        -------

        '''

        vp = []

        if points.ndim == 1:
            vp = 0
            if len(points) != 3:
                print("Error: expected 3d point")
                return
            else:
                for ii in range(self.number_electrode):
                    vp += self.electrodes[ii].field_contribution(points)

        elif points.ndim == 2:
            if points.shape[1] != 3:
                print("Error: expected 3d points")
                return
            else:

                vp = np.zeros(points.shape[0])
                for pp in range(0, len(vp)):
                    pf = 0
                    cur_point = points[pp]
                    for ii in range(self.number_electrode):
                        pf += self.electrodes[ii].field_contribution(cur_point, model=self.model)

                    vp[pp] = pf

        return vp


    def save_currents(self, filename):
        np.save(filename, self.currents)
        print('Currents saved successfully to file ', filename)


    def load_currents(self, filename):
        if os.path.isfile(filename):
            currente = np.load(filename)
            if len(currents) != self.number_electrode:
                print('Error: number of currents in file different than number of electrodes')
            else:
                print('Currents loaded successfully from file ', filename)
                self.currents(currents)
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
        normal = np.cross(rot_pos[1] - rot_pos[0], rot_pos[-1] - rot_pos[0])

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
        self.dim = info['dim']
        if isinstance(self.dim, int):
            self.dim = [self.dim, self.dim]

    # override [] method
    def __getitem__(self, index):
        # return row of current matrix
        if index < self.dim[0]:
            electrode_matrix = self.get_electrode_matrix()
            return electrode_matrix[index]
        else:
            print("Index out of bound")
            return None

    def get_electrodes_number(self):
        return self.number_electrode

    def get_current_matrix(self):
        current_matrix = np.zeros(self.dim)
        for i in range(0, self.dim[0]):
            for j in range(0, self.dim[1]):
                current_matrix[i, j] = self.currents[self.dim[0] * j + i]
        return current_matrix

    def get_electrode_matrix(self):
        electrode_matrix = np.empty(self.dim, dtype=object)
        for i in range(0, self.dim[0]):
            for j in range(0, self.dim[1]):
                electrode_matrix[i, j] = self.electrodes[self.dim[0] * j + i]
        return electrode_matrix

    def set_current_matrix(self, currents):
        # current_array = np.zeros((self.number_electrode))
        # for yy in range(self.dim):
        #     for zz in range(self.dim):
        #         current_array[self.dim * yy + zz] = currents[zz, yy]
        self.set_currents(currents=np.reshape(currents, self.dim))


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


def get_positions(elinfo):
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
        if pos.shape[1] == 2:
            pos2d = pos
            if 'plane' not in elinfo.keys():
                print("'plane' field with 2D dimensions assumed to be 'yz")
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
        if 'pitch' not in elinfo.keys():
            raise AttributeError("When 'dim' is used, also 'pitch' should be specified.")
        else:
            pitch = elinfo['pitch']

        if isinstance(dim, int):
            dim = [dim, dim]
        if isinstance(pitch, int) or isinstance(pitch, float):
            pitch = [pitch, pitch]
        if len(dim) == 2:
            d1 = np.array([])
            d2 = np.array([])
            if 'stagger' in elinfo.keys():
                stagger = elinfo['stagger']
            else:
                stagger = None
            for d_i in range(dim[1]):
                if stagger is not None:
                    if isinstance(stagger, int) or isinstance(stagger, float):
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
                print("'plane' field with 2D dimensions assumed to be 'yz")
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
                    if isinstance(stagger, int) or isinstance(stagger, float):
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
                print("'plane' field with 2D dimensions assumed to be 'yz")
                plane = 'yz'
            else:
                plane = elinfo['plane']
            if 'offset' not in elinfo.keys():
                offset = 0
            else:
                offset = elinfo['offset']
            pos = add_3dim(pos2d, plane, offset)
        electrode_pos = True

    if electrode_pos:
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
    else:
        print("Define either a list of positions 'pos' or 'dim' and 'pitch'")
        return None

def check_if_rect(elinfo):
    if 'dim' in elinfo.keys():
        dim = elinfo['dim']
        if isinstance(dim, int):
            return True
        elif isinstance(dim, list):
            if len(dim) <= 2:
                return True
        return False


def get_elcoords(xoffset, dim, pitch, electrode_name, sortlist, size, plane=None, **kwargs):
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
    # TODO redesign: more flexible --> positions should be in yaml
    if 'neuronexus-32' in electrode_name.lower():
        # calculate hexagonal order
        coldims = [10,12,10]
        if 'cut' in electrode_name.lower():
            coldims = [10,10,10]
        if sum(coldims)!=np.prod(dim):
            raise ValueError('Dimensions in Neuronexus-32-channel probe do not match.')
        zshift = -pitch[1]*(coldims[1]-1)/2.
        x = np.array([0.]*sum(coldims))
        y = np.concatenate([[-pitch[0]]*coldims[0],[0.]*coldims[1],[pitch[0]]*coldims[2]])
        z = np.concatenate((np.arange(pitch[1]/2., coldims[0]*pitch[1],pitch[1]),
                            np.arange(0.,coldims[1]*pitch[1], pitch[1]),
                            np.arange(pitch[1]/2.,coldims[2]*pitch[1], pitch[1])))+zshift
    elif 'tetrode' in electrode_name.lower():
        if plane is not None:
            if plane == 'xy':
                x = np.array([-np.sqrt(2.)*size, 0, np.sqrt(2.)*radius, 0])
                y = np.array([0, -np.sqrt(2.)*size, 0, np.sqrt(2.)*radius])
                z = np.array([0, 0, 0, 0])
            elif plane == 'yz':
                y = np.array([-np.sqrt(2.)*size, 0, np.sqrt(2.)*radius, 0])
                z = np.array([0, -np.sqrt(2.)*size, 0, np.sqrt(2.)*radius])
                x = np.array([0, 0, 0, 0])
            elif plane == 'xz':
                x = np.array([-np.sqrt(2.)*size, 0, np.sqrt(2.)*radius, 0])
                z = np.array([0, -np.sqrt(2.)*size, 0, np.sqrt(2.)*radius])
                y = np.array([0, 0, 0, 0])
        else:
            x = np.array([-np.sqrt(2.)*size, 0, np.sqrt(2.)*radius, 0])
            y = np.array([0, -np.sqrt(2.)*size, 0, np.sqrt(2.)*radius])
            z = np.array([0, 0, 0, 0])
    elif 'neuropixels' in electrode_name.lower():
        if 'v1' in electrode_name.lower():
            # checkerboard structure
            x, y, z = np.mgrid[0:1,-(dim[0]-1)/2.:dim[0]/2.:1, -(dim[1]-1)/2.:dim[1]/2.:1]
            x=x+xoffset
            yoffset = np.array([pitch[0]/4.,-pitch[0]/4.]*(dim[1]/2))
            y=np.add(y*pitch[0],yoffset) #y*pitch[0]
            z=z*pitch[1]
        elif 'v2' in electrode_name.lower():
            # no checkerboard structure
            x, y, z = np.mgrid[0:1,-(dim[0]-1)/2.:dim[0]/2.:1, -(dim[1]-1)/2.:dim[1]/2.:1]
            x=x+xoffset
            y=y*pitch[0]
            z=z*pitch[1]
        else:
            raise NotImplementedError('This version of the NeuroPixels Probe is not implemented')
    else:
        x, y, z = np.mgrid[0:1,-(dim[0]-1)/2.:dim[0]/2.:1, -(dim[1]-1)/2.:dim[1]/2.:1]
        x=x+xoffset
        y=y*pitch[0]
        z=z*pitch[1]

    el_pos = np.concatenate((np.reshape(x,(x.size,1)),
                             np.reshape(y,(y.size,1)),
                             np.reshape(z,(z.size,1))), axis = 1)
    # resort electrodes in case
    el_pos_sorted = copy.deepcopy(el_pos)
    if sortlist is not None:
        for i,si in enumerate(sortlist):
            el_pos_sorted[si] = el_pos[i]

    return el_pos_sorted

def return_mea(electrode_name=None, info=None):
    '''

    Parameters
    ----------
    electrode_name

    Returns
    -------

    '''
    import yaml
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
        pos = get_positions(elinfo)
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
    import yaml

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
                elinfo = yaml.load(meafile)
            return elinfo
        else:
            print("MEA model named %s not found" % electrode_name)
            this_dir, this_filename = os.path.split(__file__)
            electrodes = [f[:-5] for f in os.listdir(os.path.join(this_dir, "electrodes"))]
            print('Available MEA: \n', electrodes)
            return

def add_mea(mea_yaml_path):
    '''Adds the mea design defined by the yaml file in the install folder

    Parameters
    ----------
    mea_yaml_file

    Returns
    -------

    '''
    import yaml
    import shutil

    path = os.path.abspath(mea_yaml_path)

    #TODO make some checks on the validity of the file
    if os.path.isfile(mea_yaml_path):
        this_dir, this_filename = os.path.split(__file__)
        shutil.copy(path, os.path.join(this_dir, 'electrodes'))
        electrodes = [f[:-5] for f in os.listdir(os.path.join(this_dir, "electrodes"))]
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
    electrodes = [f[:-5] for f in os.listdir(os.path.join(this_dir, "electrodes"))]
    for e in electrodes:
        if e == mea_name:
            os.remove(os.path.join(this_dir, "electrodes", mea_name + '.yaml'))
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


if __name__ == '__main__':
    # test
    # elinfo = {'pos': [[10,25],[10,-5],[10,5],[10,-25]]}
    import matplotlib.pylab as plt
    elinfo = {'dim': [10, 3], 'pitch': [10, 30]}
    pos = get_positions(elinfo)

    mea = return_mea(info=elinfo)
    gpos = mea.positions
    print(pos)

    plt.plot(pos[:,1], pos[:,2], '*')
    plt.plot(gpos[:,1], gpos[:,2], '*')
    plt.axis('equal')