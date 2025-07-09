from simpeg import props
from simpeg.base.pde_simulation import with_property_mass_matrices
from simpeg.electromagnetics.frequency_domain import Simulation3DElectricField
from simpeg.electromagnetics.frequency_domain.sources import BaseFDEMSrc


class JspSource(BaseFDEMSrc):
    def __init__(
        self,
        receiver_list,
        frequency,
        pore_pressure_gradient,
        density_water=1000,
        gravity_acceleration=9.81,
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
                -1
                / (self._density_water * self._gravity_acceleration)
                * MeL
                * self._pore_pressure_gradient
            )
        else:
            raise NotImplementedError

    def s_eDeriv(self, simulation, v, adjoint=False):
        if simulation._formulation == "EB":
            # if adjoint is False:
            MeLDeriv = simulation.MeCoupling_coefficientDeriv(
                self._pore_pressure_gradient, adjoint=adjoint
            )
            return (
                -1 / (self._density_water * self._gravity_acceleration) * MeLDeriv * v
            )
        else:
            raise NotImplementedError


@with_property_mass_matrices("coupling_coefficient")
class Simulation3DElectricFieldSelfPotential(Simulation3DElectricField):

    coupling_coefficient, coupling_coefficientMap, coupling_coefficientDeriv = (
        props.Invertible("SP coupling coefficient for the source term")
    )

    def __init__(
        self,
        mesh,
        coupling_coefficient=None,
        coupling_coefficientMap=None,
        **kwargs,
    ):
        self.coupling_coefficient = coupling_coefficient
        self.coupling_coefficientMap = coupling_coefficientMap
        super().__init__(mesh=mesh, **kwargs)

    @property
    def _delete_on_model_update(self):
        """
        matrices to be deleted if the model for conductivity/resistivity is updated
        """
        toDelete = super()._delete_on_model_update
        if self.coupling_coefficientMap is not None:
            toDelete = toDelete + self._clear_on_coupling_coefficient_update
        return toDelete
