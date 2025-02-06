# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Implement many body potentials.

Triplet force classes apply a force and virial on every particle in the
simulation state commensurate with the potential energy:

.. math::

    U_\mathrm{many-body} = \frac{1}{2} \sum_{i=0}^\mathrm{N_particles-1}
                           \sum_{j \ne i}
                           \sum_{j \ne k} U(\vec{r}_{ij}, \vec{r}_{ik})

where :math:`\vec{r}_{ij} = \mathrm{minimum\_image}(\vec{r}_j - \vec{r}_i)`.
`Triplet` applies a short range cutoff for performance and assumes that both
:math:`U(\vec{r}_{ij}, \vec{r}_{ik})` and its derivatives are 0 when
:math:`r_{ij} > r_\mathrm{cut}` or :math:`r_{ik} > r_\mathrm{cut}`.

Specifically, the force :math:`\vec{F}` applied to each particle :math:`i` is:

.. math::
    \vec{F_i} =
    \begin{cases}
     -\nabla V(\vec r_{ij}, \vec r_{ik})
            & r_{ij} < r_{\mathrm{cut}}
            \land r_{ik} < r_{\mathrm{cut}} \\
        0 & \mathrm{otherwise}
    \end{cases}

The per particle energy terms are:

.. math::

    U_i = \frac{1}{2} \sum_{j \ne i}
          \sum_{j \ne k} U(\vec{r}_{ij}, \vec{r}_{ik})
          [r_{ij} < r_{\mathrm{cut}} \land r_{ik} < r_{\mathrm{cut}}]
"""

import copy
import warnings

import hoomd
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeconverter import positive_real
from hoomd.data.typeparam import TypeParameter
from hoomd.md import _md
from hoomd.md.force import Force
from hoomd.sqd_plugin import _sqd_plugin


class Triplet(Force):
    """Base class triplet force.

    `Triplet` is the base class for many-body triplet forces.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.

    .. py:attribute:: r_cut

        *r_cut* :math:`[\\mathrm{length}]`, *optional*: defaults to the value
        ``default_r_cut`` specified on construction.

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `float`])

    .. py:attribute:: nlist

        Neighbor list used to compute the triplet potential.

        Type: `hoomd.md.nlist.NeighborList`

    Warning:
        Currently HOOMD-blue does not support reverse force communication
        between MPI domains on the GPU. Since reverse force communication is
        required for the calculation of three-body forces, attempting to use
        this potential on the GPU with MPI will result in an error.
    """

    def __init__(self, nlist, default_r_cut=None):
        super().__init__()
        r_cut_param = TypeParameter(
            'r_cut', 'particle_types',
            TypeParameterDict(positive_real, len_keys=2))
        if default_r_cut is not None:
            r_cut_param.default = default_r_cut
        self._add_typeparam(r_cut_param)
        self._param_dict.update(
            ParameterDict(nlist=hoomd.md.nlist.NeighborList))
        self.nlist = nlist

    def _setattr_param(self, attr, value):
        if attr == "nlist":
            self._nlist_setter(value)
            return
        super()._setattr_param(attr, value)

    def _nlist_setter(self, new_nlist):
        if self._attached:
            raise RuntimeError("nlist cannot be set after scheduling.")
        self._param_dict._dict["nlist"] = new_nlist

    def _attach_hook(self):
        if self.nlist._attached and self.nlist._simulation != self._simulation:
            warnings.warn(
                f"{self} object is creating a new equivalent neighbor list."
                
                f" This is happending since the force is moving to a new "
                f"simulation. Explicitly set the nlist to hide this warning.",
                RuntimeWarning)
            self.nlist = copy.deepcopy(self.nlist)
        self.nlist._attach(self._simulation)
        self.nlist._cpp_obj.setStorageMode(_md.NeighborList.storageMode.full)
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = getattr(_sqd_plugin, self._cpp_class_name)
        else:
            cls = getattr(_sqd_plugin, self._cpp_class_name + "GPU")

        self._cpp_obj = cls(self._simulation.state._cpp_sys_def,
                            self.nlist._cpp_obj)


class SquareDensity(Triplet):
    r"""Soft force for simulating a van der Waals liquid.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.

    `SquareDensity` that the square density three-body force should on every
    particle in the simulation state.

    The self energy per particle takes the form:

    .. math::

        \Psi^{ex} = B (\rho - A)^2

    which gives a pair-wise additive, three-body force:

    .. math::

        \vec{f}_{ij} = \left( B (n_i - A) + B (n_j - A) \right) w'_{ij}
        \vec{e}_{ij}

    Here, :math:`w_{ij}` is a quadratic, normalized weighting function,

    .. math::

        w(x) = \frac{15}{2 \pi r_{c,\mathrm{weight}}^3}
               (1-r/r_{c,\mathrm{weight}})^2

    The local density at the location of particle *i* is defined as

    .. math::

        n_i = \sum\limits_{j\neq i} w_{ij}
              \left(\big| \vec r_i - \vec r_j \big|\right)

    .. py:attribute:: params

        The SquareDensity potential parameters. The dictionary has the
        following keys:

        * ``A`` (`float`, **required**) - :math:`A` - mean density
          (*default*:0) :math:`[\mathrm{length}^{-2}]` in 2D and
          :math:`[\mathrm{length}^{-3}]` in 3D
        * ``B`` (`float`, **required**) - :math:`B` - coefficient of the
          harmonic density term
          :math:`[\mathrm{energy} \cdot \mathrm{length}^4]` in 2D and
          :math:`[\mathrm{energy} \cdot \mathrm{length}^6]` in 3D

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = nlist.Cell()
        sqd = md.many_body.SquareDensity(nl, default_r_cut=3.0)
        sqd.params[('A', 'B')] = dict(A=1.0, B=2.0)
        sqd.params[('B', 'B')] = dict(A=2.0, B=2.0, default_r_on=1.0)

    For further details regarding this multibody potential, see

    [1] P. B. Warren, "Vapor-liquid coexistence in many-body dissipative
    particle dynamics" Phys. Rev. E. Stat. Nonlin. Soft Matter Phys., vol. 68,
    no. 6 Pt 2, p. 066702, 2003.
    """
    _cpp_class_name = "PotentialSquareDensity2"

    def __init__(self, nlist, default_r_cut=None):
        super().__init__(nlist, default_r_cut)
        params = TypeParameter('params', 'particle_types',
                               TypeParameterDict(A=0.0, B=float, attr_rcut=1.0, rep_rcut=1.0, density_rcut=1.0, ncut_ij=1.0, ncut_ji=1.0, len_keys=2))
        self._add_typeparam(params)