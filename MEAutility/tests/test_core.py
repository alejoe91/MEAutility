import pytest
import numpy as np
from MEAutility.core import Electrode
import MEAutility as mu
import yaml
import os


def test_electrodes_field_contribution():
    elec_c = Electrode(position=[100,100,100], normal=[1,0,0], current=10, size=10, shape='circle')
    v_1_c, _ = elec_c.field_contribution(pos=[150,100,100], npoints=1, main_axes=[[0,1,0],[0,0,1]])
    v_5_c, _ = elec_c.field_contribution(pos=[150, 100, 100], npoints=5, main_axes=[[0,1,0],[0,0,1]])
    v_10_c, _ = elec_c.field_contribution(pos=[150, 100, 100], npoints=10, main_axes=[[0,1,0],[0,0,1]])

    elec_s = Electrode(position=[100, 100, 100], normal=[1, 0, 0], current=10, size=10, shape='square')
    v_1_s, _ = elec_s.field_contribution(pos=[150, 100, 100], npoints=1, main_axes=[[0, 1, 0], [0, 0, 1]])
    v_5_s, _ = elec_s.field_contribution(pos=[150, 100, 100], npoints=5, main_axes=[[0, 1, 0], [0, 0, 1]])
    v_10_s, _ = elec_s.field_contribution(pos=[150, 100, 100], npoints=10, main_axes=[[0, 1, 0], [0, 0, 1]])

    assert np.isclose([v_1_c], [v_10_c], rtol=0.1)
    assert np.isclose([v_1_c], [v_5_c], rtol=0.1)
    assert np.isclose([v_10_c], [v_5_c], rtol=0.1)

    assert np.isclose([v_1_s], [v_10_s], rtol=0.1)
    assert np.isclose([v_1_s], [v_5_s], rtol=0.1)
    assert np.isclose([v_10_s], [v_5_s], rtol=0.1)


def test_return_mea():
    mea = mu.return_mea('Neuronexus-32')
    assert isinstance(mea, mu.core.MEA)
    assert len(mea.electrodes) == 32
    assert len(mea.positions) == 32
    assert mea.model == 'semi'
    assert mea.plane == 'yz'
    assert mea.type == 'mea'
    assert mea.size == 7.5


def test_mea_set_currents():
    mea = mu.return_mea('Neuronexus-32')
    mea.currents = np.arange(32)
    assert np.isclose(mea.currents, np.arange(32)).all()

    mea.currents = np.zeros((32, 200))
    assert len(mea.currents[0]) == 200
    assert mea.currents.shape == (32, 200)


def test_mea_compute_field():
    mea = mu.return_mea('Neuronexus-32')
    pos = mea.positions[10]
    mea.set_current(10, 100)
    dist = 30
    pos_c = pos + [dist, 0, 0]

    v = 100 / (2*np.pi*0.3 * dist)
    v_c = mea.compute_field(pos_c)
    assert v == v_c

    v_arr = np.array([v] * 100)
    mea.set_current(10, [100]*100)
    v_c_arr = mea.compute_field(pos_c)

    assert np.isclose(v_arr,v_c_arr).all()


def test_mea_save_load():
    mea = mu.return_mea('Neuronexus-32')
    mea.currents = np.arange(32)
    mea.save_currents('currents')

    mea_loaded = mu.return_mea('Neuronexus-32')
    mea_loaded.load_currents('currents.npy')
    os.remove('currents.npy')
    assert np.isclose(mea.currents, mea_loaded.currents).all()


def test_mea_handling():
    mea = mu.return_mea('Neuronexus-32')
    new_center = [100, 300, 500]
    mea.move(new_center)

    assert np.isclose(np.mean(mea.positions, axis=0), new_center).all()
    mea.center()
    assert np.isclose(np.mean(mea.positions, axis=0), [0, 0, 0]).all()
    mea.rotate([0, 0, 1], 90)
    assert np.isclose(mea.positions[:, 1], np.zeros(len(mea.positions))).all()


def test_add_remove_list_mea():
    info = mu.return_mea_info('Neuronexus-32')
    test_name = 'Neuronexus-test'
    info['electrode_name'] = test_name

    with open(test_name + '.yaml', 'w') as f:
        yaml.dump(info, f)
    number_mea = len(mu.return_mea_list())
    mu.add_mea(test_name + '.yaml')
    os.remove(test_name + '.yaml')
    assert len(mu.return_mea_list()) == number_mea + 1
    assert test_name in  mu.return_mea_list()
    mu.remove_mea(test_name)
    assert len(mu.return_mea_list()) == number_mea
    assert test_name not in mu.return_mea_list()


def test_rectmea():
    mea = mu.return_mea('SqMEA-10-15')
    mea.currents = np.arange(100)

    assert mea[9][0].current == mea.currents[9]
    assert mea[0][1].current == mea.currents[10]

    current_img = mea.get_current_matrix()
    assert current_img.shape == (10, 10)

    new_currents = np.ones((10,10))
    mea.set_current_matrix(new_currents)
    assert np.isclose(new_currents, mea.get_current_matrix()).all()