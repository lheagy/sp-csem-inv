import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from string import ascii_lowercase
import os

import discretize

from simpeg import (
    maps,
    utils,
)

import spcsem
from simpeg.electromagnetics import frequency_domain as fdem

Solver = utils.solver_utils.get_default_solver()

# show plots
PLOTIT = False

# testing directory
this_dir = os.path.dirname(__file__)
E3DDIR = os.path.sep.join([this_dir, "..", "e3d"])

# load mesh, conductivity model, data
mesh = discretize.TreeMesh.read_UBC(
    os.path.sep.join([E3DDIR, "J", "octree_mesh_casing.txt"])
)
conductivity_model = discretize.TreeMesh.read_model_UBC(
    mesh, os.path.sep.join([E3DDIR, "J", "model_casing.con"])
)
data_locs = np.loadtxt(os.path.sep.join([E3DDIR, "points", "points.txt"]))
data_e3d = np.loadtxt(os.path.sep.join([E3DDIR, "points", "dpredFWD.txt"]))

# parameters used for the E3D simulation
gravity_acceleration = 9.81
density_water = 1000.0
coupling_coefficient = 1e-5
frequency = 1
air_earth_interface = 500


def plot_currents(mesh, jx, jy, jz):
    fig, ax = plt.subplots(1, 3, figsize=(12, 2))

    for a, j, comp in zip(ax, [jx, jy, jz], ["x", "y", "z"]):
        plt.colorbar(
            mesh.plot_slice(
                getattr(mesh, f"average_edge_{comp}_to_cell") * j,
                ax=a,
                normal="y",
                pcolor_opts={
                    "norm": SymLogNorm(vmin=-1e-2, vmax=1e-2, linthresh=1e-6),
                    "cmap": "RdBu_r",
                },
            )[0],
            ax=a,
        )
        a.set_xlim(np.r_[-1, 1] * 400)
        a.set_ylim(np.r_[0, 550])
        a.set_aspect(1)


def plot_data(rx_x, rx_y, data_x_re, data_x_im, data_y_re, data_y_im):
    fig, ax = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)

    data_x = np.vstack([data_x_re, data_x_im]).T
    data_y = np.vstack([data_y_re, data_y_im]).T

    for i, data_plot in enumerate([data_x, data_y]):
        for j, comp in enumerate(["real", "imag"]):
            plt.colorbar(
                ax[i, j].contourf(
                    rx_x,
                    rx_y,
                    data_plot[:, j].reshape(len(rx_x), len(rx_y), order="F"),
                    levels=50,
                    cmap="jet",
                ),
                ax=ax[i, j],
            )
            ax[i, j].set_title(f"E{ascii_lowercase[23+i]} {comp}")
    for a in ax.flatten():
        a.set_aspect(1)


def generate_survey():
    jx_e3d = np.loadtxt(os.path.sep.join([E3DDIR, "J", "J_X.txt"]))
    jy_e3d = np.loadtxt(os.path.sep.join([E3DDIR, "J", "J_Y.txt"]))
    jz_e3d = np.loadtxt(os.path.sep.join([E3DDIR, "J", "J_Z.txt"]))

    jx = spcsem.utils.match_values_nearest(jx_e3d, mesh.edges_x)
    jy = spcsem.utils.match_values_nearest(jy_e3d, mesh.edges_y)
    jz = spcsem.utils.match_values_nearest(jz_e3d, mesh.edges_z)

    if PLOTIT is True:
        plot_currents(mesh, jx, jy, jz)

    jsrc = -np.hstack(
        [jx, jy, jz]
    )  # note negative b/c z is positive up in SimPEG and positive down in E3D

    rx_list = [
        fdem.receivers.PointElectricField(
            locations=data_locs, orientation="x", component="real"
        ),
        fdem.receivers.PointElectricField(
            locations=data_locs, orientation="x", component="imag"
        ),
        fdem.receivers.PointElectricField(
            locations=data_locs, orientation="y", component="real"
        ),
        fdem.receivers.PointElectricField(
            locations=data_locs, orientation="y", component="imag"
        ),
    ]

    source = spcsem.JspSource(
        receiver_list=rx_list,
        frequency=frequency,
        pore_pressure_gradient=1
        / (-coupling_coefficient / (density_water * gravity_acceleration))
        * jsrc,
        density_water=density_water,
        gravity_acceleration=gravity_acceleration,
    )

    survey = fdem.Survey([source])
    return survey


def generate_simulation():
    survey = generate_survey()

    active_cells = (
        mesh.cell_centers[:, 2] < air_earth_interface,
    )  # all cells below 500m are active
    coupling_coefficient_map = maps.InjectActiveCells(
        mesh=mesh,
        active_cells=active_cells,
        value_inactive=0,  # set coupling coefficient to 0 in air
    )

    sim = spcsem.Simulation3DElectricFieldSelfPotential(
        mesh=mesh,
        survey=survey,
        solver=Solver,
        sigma=conductivity_model,
        coupling_coefficientMap=coupling_coefficient_map,
        storeJ=True,
    )
    return active_cells, sim


def test_forward():
    active_cells, sim = generate_simulation()
    dpred = sim.make_synthetic_data(
        coupling_coefficient * np.ones(np.sum(active_cells))
    )

    data_e3d_x = data_e3d[::2, :]
    data_e3d_y = data_e3d[1::2, :]

    source = sim.survey.source_list[0]
    rx_list = source.receiver_list

    if PLOTIT:
        rx_x = np.unique(source.receiver_list[0].locations[:, 0])
        rx_y = np.unique(source.receiver_list[0].locations[:, 1])
        plot_data(
            rx_x,
            rx_y,
            data_e3d_x[:, 0],
            data_e3d_x[:, 1],
            data_e3d_y[:, 0],
            data_e3d_y[:, 1],
        )
        plot_data(
            rx_x,
            rx_y,
            dpred[source, rx_list[0]],
            dpred[source, rx_list[1]],
            dpred[source, rx_list[3]],
            dpred[source, rx_list[4]],
        )

    # note that SimPEG uses +i\omega t while e3d uses e -i\omega t
    max_re = np.max(np.abs(data_e3d[:, 0]))
    max_im = np.max(np.abs(data_e3d[:, 1]))
    np.allclose(
        data_e3d_x[:, 0], dpred[source, rx_list[0]], atol=1e-3 * max_re, rtol=0.03
    )
    np.allclose(
        -data_e3d_x[:, 1], dpred[source, rx_list[1]], atol=1e-2 * max_im, rtol=0.03
    )
    np.allclose(
        data_e3d_y[:, 0], dpred[source, rx_list[2]], atol=1e-3 * max_re, rtol=0.03
    )
    np.allclose(
        -data_e3d_y[:, 1], dpred[source, rx_list[3]], atol=1e-2 * max_im, rtol=0.03
    )


def test_sensitivities():
    active_cells, sim = generate_simulation()

    x0 = np.log(1e-5 + 1e-7 * np.random.randn(np.sum(active_cells)))

    def fun(x):
        return sim.dpred(x), lambda x: sim.Jvec(x0, x)

    discretize.tests.check_derivative(fun, x0, num=3)


if __name__ == "__main__":
    test_forward()
    test_sensitivities()
