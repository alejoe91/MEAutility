import pytest
import numpy as np
import MEAutility as mu


def test_plot_probe():
    mea = mu.return_mea('Neuronexus-32')
    ax = mu.plot_probe(mea)


def test_plot_probe_3d():
    mea = mu.return_mea('Neuronexus-32')
    ax = mu.plot_probe_3d(mea)
    mea = mu.return_mea('SqMEA-10-15')
    ax = mu.plot_probe_3d(mea, ax=ax, type='planar', color_currents=True)


def test_plot_v_image():
    mea = mu.return_mea('Neuronexus-32')
    ax = mu.plot_v_image(mea, offset=30, y_bound=[-100, 100], z_bound=[-100, 100])


def test_plot_v_surf():
    mea = mu.return_mea('Neuronexus-32')
    ax = mu.plot_v_surf(mea, offset=30, y_bound=[-100, 100], z_bound=[-100, 100])


def test_plot_mea_recording():
    mea = mu.return_mea('Neuronexus-32')
    rec = np.random.randn(32, 500)
    ax = mu.plot_mea_recording(rec, mea)


def test_play_mea_recording():
    mea = mu.return_mea('Neuronexus-32')
    rec = np.random.randn(32, 500)
    ax = mu.play_mea_recording(rec, mea, fs=32000)