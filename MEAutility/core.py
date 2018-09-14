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
    def __init__(self, position=None, current=False, sigma=None, points_per_edge=None, shape='square', side_length=None):
        '''

        Parameters
        ----------
        position
        current
        sigma
        points_per_edge
        shape
        side_length
        '''
        # convert to numpy array
        if type(position) == list:
            position = np.array(position)

        self.position = position
        if sigma:
            self.sigma = sigma
        else:
            self.sigma = 0.3

        self.max_field = 1000

        if current:
            self.current = current
        else:
            self.current = 0

        if points_per_edge:
            self.points_per_edge = points_per_edge
        else:
            self.points_per_edge = 1

    def set_current(self, current):
        self.current = current

    def set_sigma(self, sigma):
        self.sigma = sigma

    def set_max_field(self, max_field):
        self.max_field = max_field

    def field_contribution(self, point):
        if self.points_per_edge == 1:
            if any(point != self.position):
                return self.current / (2*np.pi*self.sigma*la.norm(point-self.position))
            else:
                print("WARNING: point and electrode location are coincident! Field set to MAX_FIELD: ", self.max_field)
                return self.max_field
        # add return of x-y-z components of the field


class MEA(object):
    '''

    '''
    def __init__(self, electrodes=None, dim=None, pitch=None):
        '''

        Parameters
        ----------
        electrodes
        '''
        if electrodes is not None:
            self.electrodes = electrodes
            if type(electrodes) in (np.ndarray, list):
                self.number_electrode = len(electrodes)
                # print("Create MEA with ", len(self.electrodes), " electrodes")
            elif isinstance(electrodes, Electrode):
                self.number_electrode = 1
                # print("Create MEA with 1 electrode")
            else:
                self.number_electrode = 0
                print("Wrong arguments")
        else:
            self.number_electrode = 0
            # print("Created empty MEA: add electrodes!")

        self.dim = dim
        self.pitch = pitch

    def set_electrodes(self, electrodes):
        self.electrodes = electrodes
        self.number_electrode = len(electrodes)

    def set_currents(self, currents_array):
        for ii in range(self.number_electrode):
            self.electrodes[ii].set_current(currents_array[ii])

    def set_random_currents(self, amp=None):
        if amp:
            currents = np.random.randn(self.number_electrode) * amp
        else:
            currents = np.random.randn(self.number_electrode) * 10

        for ii in range(self.number_electrode):
            self.electrodes[ii].set_current(currents[ii])

    def reset_currents(self, amp=None):
        currents = np.zeros(self.number_electrode)

        for ii in range(self.number_electrode):
            self.electrodes[ii].set_current(currents[ii])
    
    def get_currents(self):
        currents = np.zeros(self.number_electrode)
        for ii in range(self.number_electrode):
            currents[ii] = self.electrodes[ii].current
        return currents

    def get_electrode_array(self):
        return self.electrodes

    def get_electrode_positions(self):
        pos = np.zeros((self.number_electrode,3))
        for ii in range(self.number_electrode):
            pos[ii, :] = self.electrodes[ii].position
        return pos

    def compute_field(self, points):

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
                        pf += self.electrodes[ii].field_contribution(cur_point)

                    vp[pp] = pf

        return vp

    def save_currents(self, filename):
        #todo save numpy
        # with open(filename, 'w') as f:
        #     for a in range(self.number_electrode):
        #         print >> f, "%g" % (self.get_currents()[a])

        print('Currents saved successfully to file ', filename)

    def load_currents(self, filename):
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                currents = []
                for line in f:
                    currents.append(int(line))

                if len(currents) != self.number_electrode:
                    print('Error: number of currents in file different than number of electrodes')
                else:
                    print('Currents loaded successfully from file ', f.name)
                    self.set_currents(currents)
        else:
            print('File does not exist')


class SquareMEA(MEA):
    '''

    '''
    def __init__(self, dim=None, pitch=None, width=None, x_plane=None):
        '''

        Parameters
        ----------
        dim
        pitch
        width
        x_plane
        '''
        MEA.__init__(self)

        if width:
            if pitch:
                self.dim = width // pitch
                self.pitch = pitch
            elif dim:
                self.pitch = width // dim
                self.dim = dim
            else:
                raise AttributeError('If width is specified, either dim or pitch must be set as well')
        else:
            if pitch:
                self.pitch = pitch
            else:
                self.pitch = 15
            if dim:
                self.dim = dim
            else:
                self.dim = 10

        if x_plane:
            self.x_plane = x_plane
        else:
            self.x_plane = 0

        # Create matrix of electrodes
        if (self.dim % 2 is 0):
            # print(self.dim, 'even self.dim')
            sources_pos_y = range(-self.dim / 2 * self.pitch + self.pitch / 2, self.dim / 2 * self.pitch, self.pitch)
        else:
            sources_pos_y = range(-(self.dim / 2) * self.pitch, (self.dim / 2) * self.pitch + self.pitch, self.pitch)
        sources_pos_z = sources_pos_y[::-1]
        sources = []
        for ii in sources_pos_y:
            for jj in sources_pos_z:
                # list converted to np.array in Electrode constructor
                sources.append(Electrode([self.x_plane, ii, jj]))

        MEA.set_electrodes(self, sources)

    # override [] method
    def __getitem__(self, index):
        # return row of current matrix
        if index < self.dim:
            electrode_matrix = self.get_electrode_matrix()
            return electrode_matrix[index]
        else:
            print("Index out of bound")
            return None

    def get_electrodes_number(self):
        return self.dim**2

    def get_current_matrix(self):
        current_matrix = np.zeros((self.dim, self.dim))
        for yy in range(self.dim):
            for zz in range(self.dim):
                current_matrix[zz, yy] = MEA.get_currents(self)[self.dim * yy + zz]
        return current_matrix

    def get_electrode_matrix(self):
        electrode_matrix = [[0 for x in range(self.dim)] for y in range(self.dim)]
        for yy in range(0, self.dim):
            for zz in range(0, self.dim):
                electrode_matrix[zz][yy] = self.electrodes[self.dim * yy + zz]
        return electrode_matrix

    def set_current_matrix(self, currents):
        current_array = np.zeros((self.number_electrode))
        for yy in range(self.dim):
            for zz in range(self.dim):
                current_array[self.dim * yy + zz] = currents[zz, yy]
        MEA.set_currents(self, currents_array=current_array)



def get_elcoords(xoffset, dim, pitch, electrode_name, sortlist, radius, plane=None, **kwargs):
    '''

    Parameters
    ----------
    xoffset
    dim
    pitch
    electrode_name
    sortlist
    kwargs

    Returns
    -------

    '''
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
                x = np.array([-np.sqrt(2.)*radius, 0, np.sqrt(2.)*radius, 0])
                y = np.array([0, -np.sqrt(2.)*radius, 0, np.sqrt(2.)*radius])
                z = np.array([0, 0, 0, 0])
            elif plane == 'yz':
                y = np.array([-np.sqrt(2.)*radius, 0, np.sqrt(2.)*radius, 0])
                z = np.array([0, -np.sqrt(2.)*radius, 0, np.sqrt(2.)*radius])
                x = np.array([0, 0, 0, 0])
            elif plane == 'xz':
                x = np.array([-np.sqrt(2.)*radius, 0, np.sqrt(2.)*radius, 0])
                z = np.array([0, -np.sqrt(2.)*radius, 0, np.sqrt(2.)*radius])
                y = np.array([0, 0, 0, 0])
        else:
            x = np.array([-np.sqrt(2.)*radius, 0, np.sqrt(2.)*radius, 0])
            y = np.array([0, -np.sqrt(2.)*radius, 0, np.sqrt(2.)*radius])
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


def return_mea(electrode_name=None, x_plane=None, **kwargs):
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
        with open(os.path.join(electrode_path, electrode_name + '.yaml')) as meafile:
            elinfo = yaml.load(meafile)

        if x_plane is None:
            x_plane = 0.
        if 'sortlist' in kwargs.keys():
            sortlist = kwargs['sortlist']
            if sortlist == None:
                elinfo['sortlist'] = None
            pos = get_elcoords(x_plane,**elinfo)
        else:
            pos = get_elcoords(x_plane, **elinfo)

        return pos, elinfo['dim'], elinfo['pitch']

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
        with open(os.path.join(electrode_path, electrode_name + '.yaml')) as meafile:
            elinfo = yaml.load(meafile)

        return elinfo