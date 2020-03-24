import pytest
import numpy as np
from MEAutility.core import Electrode
import MEAutility as mu
import yaml
import os


def test_electrodes_field_contribution():
    elec_c = Electrode(position=[100, 100, 100], normal=[1, 0, 0], current=10, size=10, shape='circle',
                       main_axes=[[0, 1, 0], [0, 0, 1]])
    v_1_c = elec_c.field_contribution(points=[150, 100, 100], npoints=1)
    v_5_c = elec_c.field_contribution(points=[150, 100, 100], npoints=5)
    v_10_c = elec_c.field_contribution(points=[150, 100, 100], npoints=10)

    elec_s = Electrode(position=[100, 100, 100], normal=[1, 0, 0], current=10, size=10, shape='square',
                       main_axes=[[0, 1, 0], [0, 0, 1]])
    v_1_s = elec_s.field_contribution(points=[150, 100, 100], npoints=1)
    v_5_s = elec_s.field_contribution(points=[150, 100, 100], npoints=5)
    v_10_s = elec_s.field_contribution(points=[150, 100, 100], npoints=10)

    elec_r = Electrode(position=[100, 100, 100], normal=[1, 0, 0], current=10, size=[10, 20], shape='rect',
                       main_axes=[[0, 1, 0], [0, 0, 1]])
    v_1_r = elec_r.field_contribution(points=[150, 100, 100], npoints=1)
    v_5_r = elec_r.field_contribution(points=[150, 100, 100], npoints=5)
    v_10_r = elec_r.field_contribution(points=[150, 100, 100], npoints=10)

    assert np.isclose([v_1_c], [v_10_c], rtol=0.1)
    assert np.isclose([v_1_c], [v_5_c], rtol=0.1)
    assert np.isclose([v_10_c], [v_5_c], rtol=0.1)

    assert np.isclose([v_1_s], [v_10_s], rtol=0.1)
    assert np.isclose([v_1_s], [v_5_s], rtol=0.1)
    assert np.isclose([v_10_s], [v_5_s], rtol=0.1)

    assert np.isclose([v_1_r], [v_10_r], rtol=0.1)
    assert np.isclose([v_1_r], [v_5_r], rtol=0.1)
    assert np.isclose([v_10_r], [v_5_r], rtol=0.1)


def test_electrodes_field_contribution_anisotropic():
    elec_c = Electrode(position=[100, 100, 100], normal=[1, 0, 0], current=10, size=10, shape='circle',
                       sigma=[0.3, 0.4, 0.5], main_axes=[[0, 1, 0], [0, 0, 1]])
    v_1_c = elec_c.field_contribution(points=[150, 100, 100], npoints=1)
    v_5_c = elec_c.field_contribution(points=[150, 100, 100], npoints=5)
    v_10_c = elec_c.field_contribution(points=[150, 100, 100], npoints=10)

    elec_s = Electrode(position=[100, 100, 100], normal=[1, 0, 0], current=10, size=10, shape='square',
                       sigma=[0.3, 0.4, 0.5], main_axes=[[0, 1, 0], [0, 0, 1]])
    v_1_s = elec_s.field_contribution(points=[150, 100, 100], npoints=1)
    v_5_s = elec_s.field_contribution(points=[150, 100, 100], npoints=5)
    v_10_s = elec_s.field_contribution(points=[150, 100, 100], npoints=10)

    elec_r = Electrode(position=[100, 100, 100], normal=[1, 0, 0], current=10, size=[10, 20], shape='rect',
                       sigma=[0.3, 0.4, 0.5], main_axes=[[0, 1, 0], [0, 0, 1]])
    v_1_r = elec_r.field_contribution(points=[150, 100, 100], npoints=1)
    v_5_r = elec_r.field_contribution(points=[150, 100, 100], npoints=5)
    v_10_r = elec_r.field_contribution(points=[150, 100, 100], npoints=10)

    assert np.isclose([v_1_c], [v_10_c], rtol=0.1)
    assert np.isclose([v_1_c], [v_5_c], rtol=0.1)
    assert np.isclose([v_10_c], [v_5_c], rtol=0.1)

    assert np.isclose([v_1_s], [v_10_s], rtol=0.1)
    assert np.isclose([v_1_s], [v_5_s], rtol=0.1)
    assert np.isclose([v_10_s], [v_5_s], rtol=0.1)

    assert np.isclose([v_1_r], [v_10_r], rtol=0.1)
    assert np.isclose([v_1_r], [v_5_r], rtol=0.1)
    assert np.isclose([v_10_r], [v_5_r], rtol=0.1)


def test_return_mea():
    mea = mu.return_mea('Neuronexus-32')
    assert isinstance(mea, mu.core.MEA)
    assert len(mea.electrodes) == 32
    assert len(mea.positions) == 32
    assert mea.model == 'semi'
    assert mea.plane == 'yz'
    assert mea.type == 'mea'
    assert mea.size == 7.5


def test_get_n_points():
    mea = mu.return_mea('Neuronexus-32')
    points = mea.get_random_points_inside(20)
    for (pos, p) in zip(mea.positions, points):
        assert np.all([np.linalg.norm(p_i - pos) <= mea.size for p_i in p])


def test_mapping():
    mea = mu.return_mea('Neuronexus-32')
    mapping = np.random.randn(mea.number_electrodes, 1000)
    points = np.random.randn(1000, 3)

    mea.set_mapping(mapping, points)

    for i, el in enumerate(mea.electrodes):
        assert np.allclose(mapping[i], np.squeeze(el.mapping))


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

    v = 100 / (2 * np.pi * 0.3 * dist)
    v_c = mea.compute_field(pos_c)
    assert np.isclose(v, v_c)

    v_arr = np.array([v] * 100)
    mea.set_current(10, [100] * 100)
    v_c_arr = mea.compute_field(pos_c)

    assert np.isclose(v_arr, v_c_arr).all()


def test_mea_set_current_pulse():
    mea = mu.return_mea('Neuronexus-32')
    el_id = 0
    amp1 = 1000
    t_stop = 100
    t_start = 5
    width1 = 2
    interpulse = 2
    dt = 0.01
    n_pulses = 3
    interburst = 30

    c, t = mea.set_current_pulses(el_id=0, amp1=amp1, width1=width1, interpulse=interpulse, t_stop=t_stop, dt=dt,
                                  biphasic=False)
    assert np.max(t) < t_stop
    assert np.max(c) == amp1 and np.min(c) == 0
    assert np.allclose(c, mea.electrodes[el_id].current)

    c, t = mea.set_current_pulses(el_id=0, amp1=amp1, width1=width1, interpulse=interpulse, t_stop=t_stop, dt=dt,
                                  biphasic=True)
    assert np.max(t) < t_stop
    assert np.max(c) == amp1 and np.min(c) == -amp1
    assert np.allclose(c, mea.electrodes[el_id].current)

    c, t = mea.set_current_pulses(el_id=0, amp1=amp1, width1=width1, interpulse=interpulse, t_stop=t_stop, dt=dt,
                                  biphasic=True, n_pulses=n_pulses, interburst=interburst, t_start=t_start)
    assert np.max(t) < t_stop and np.min(t) >= t_start
    assert np.max(c) == amp1 and np.min(c) == -amp1
    assert np.allclose(c, mea.electrodes[el_id].current)


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
    assert test_name in mu.return_mea_list()
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

    new_currents = np.ones((10, 10))
    mea.set_current_matrix(new_currents)
    assert np.isclose(new_currents, mea.get_current_matrix()).all()
