import numpy as np

from simpeg import props
from simpeg.base.pde_simulation import with_property_mass_matrices
from simpeg.electromagnetics.frequency_domain import Simulation3DElectricField
from simpeg.electromagnetics.frequency_domain.sources import BaseFDEMSrc

class JspSource(BaseFDEMSrc):
    def __init__(
            self, receiver_list, frequency, pore_pressure_gradient,
            density_water=1000, gravity_acceleration=9.81
        ):
            self._density_water = density_water
            self._gravity_acceleration = gravity_acceleration
            self._pore_pressure_gradient = pore_pressure_gradient
            super().__init__(
                 receiver_list=receiver_list, frequency=frequency, integrate=True
            )

    def s_e(self, simulation):
        """Electric source term (s_e)

        Parameters
        ----------
        simulation : BaseFDEMSimulation
            SimPEG FDEM simulation

        Returns
        -------
        numpy.ndarray
            electric source term on mesh.
        """
        if simulation._formulation == "EB":
            MeL = simulation.MeCoupling_coefficient
            return (
                - 1/(self._density_water * self._gravity_acceleration) *
                MeL *
                self._pore_pressure_gradient
            )
        # simulation.Me * self._s_e
        else:
             raise NotImplementedError

    def s_eDeriv(self, simulation, v, adjoint=False):
        if simulation._formulation == "EB":
            # if adjoint is False:
            MeLDeriv = simulation.MeCoupling_coefficientDeriv(self._pore_pressure_gradient, adjoint=adjoint)
            return (
                - 1/(self._density_water * self._gravity_acceleration) *
                MeLDeriv * v
            )
            # elif adjoint is True:
            #     MeLDerivT = simulation.MeCoupling_coefficientDeriv(v, adjoint=True)
            #     return (
            #         - 1/(self._density_water * self._gravity_acceleration) *
            #         MeLDerivT * self._pore_pressure_gradient
            #     )


        # simulation.Me * self._s_e
        else:
            raise NotImplementedError


@with_property_mass_matrices("coupling_coefficient")
class Simulation3DElectricFieldSelfPotential(Simulation3DElectricField):

    coupling_coefficient, coupling_coefficient_map, coupling_coefficient_deriv = props.Invertible("Electrical resistivity (Ohm m)")
