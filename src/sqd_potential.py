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


class Tersoff(Triplet):
    r"""Tersoff force.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list
        default_r_cut (float): Default cutoff radius :math:`[\mathrm{length}]`.

    The Tersoff potential is a bond-order potential based on the Morse potential
    that accounts for the weakening of individual bonds with increasing
    coordination number. It does this by computing a modifier to the attractive
    term of the potential. The modifier contains the effects of third-bodies on
    the bond energies. The potential also includes a smoothing function around
    the cutoff. The implemented smoothing function is exponential in nature as
    opposed to the sinusoid used by `J. Tersoff 1988`_.

    .. _J. Tersoff 1988:
      https://journals.aps.org/prb/abstract/10.1103/PhysRevB.38.9902

    `Tersoff` computes the Tersoff three-body force on every particle in the
    simulation state. Despite the fact that the Tersoff potential accounts for
    the effects of third bodies, the species of the third body is irrelevant. It
    can thus use type-pair parameters similar to those of the pair potentials:

    .. math::

        U_{ij}(r) = \frac{1}{2} f_C(r_{ij})
            \left[f_R(r_{ij}) + b_{ij}f_A(r_{ij})\right]

    where

    .. math::

        f_R(r) = A_1e^{\lambda_1(r_D-r)}

        f_A(r) = A_2e^{\lambda_2(r_D-r)}

    .. math::
        f_C(r) =
        \begin{cases}
            1 & r < r_{\mathrm{cut}} - r_{CT} \\
            \exp \left[-\alpha\frac{x(r)^3}{x(r)^3 - 1} \right]
                & r_{\mathrm{cut}} - r_{CT} < r < r_{\mathrm{cut}} \\
            0 & r > r_{\mathrm{cut}}
        \end{cases}

    .. math::

        b_{ij} = (1 + \gamma^n\chi_{ij}^n)^{\frac{-1}{2n}}

    In the definition of :math:`f_C(r)`, there is a quantity :math:`x(r)`, which
    is defined as

    .. math::

        x(r) = \frac{r - (r_{\mathrm{cut}} - r_{CT})}{r_{CT}}

    which ensures continuity between the different regions of the potential. In
    the definition of :math:`b_{ij}`, there is a quantity :math:`\chi_{ij}`
    which is defined as

    .. math::

        \chi_{ij} = \sum_{k \neq i,j} f_C(r_{ik})
                     \cdot e^{\lambda_3^3 |r_{ij} - r_{ik}|^3}
        \cdot g(\theta_{ijk})

        g(\theta_{ijk}) = 1 + \frac{c^2}{d^2}
                           - \frac{c^2}{d^2 + |m - \cos(\theta_{ijk})|^2}

    .. py:attribute:: params

        The Tersoff potential parameters. The dictionary has the following
        keys:

        * ``magnitudes`` (tuple[`float`, `float`]) - :math:`(A_1, A_2)` -
          Magnitudes of the repulsive and attractive
          terms (*default*: (1.0, 1.0)) :math:`[\mathrm{energy}]`
        * ``exp_factors`` (tuple[`float`, `float`]) -
          :math:`(\lambda_1, \lambda_2)` - exponential factors of the
          repulsive and attractive
          terms (*default*: 2.0) :math:`[\mathrm{length}^{-1}]`
        * ``lambda3`` (`float`) - :math:`\lambda_3` - exponential factor in
          :math:`\chi_{ij}` (*default*: 0.0) :math:`[\mathrm{length}^{-1}]`
        * ``dimer_r`` (`float`) - :math:`r_D` - length shift in attractive
          and repulsive terms (*default*: 1.5) :math:`[\mathrm{length}]`
        * ``cutoff_thickness`` (`float`) - :math:`r_{CT}` - distance which
          defines the different regions of the
          potential (*default*: 0.2) :math:`[\mathrm{length}]`
        * ``alpha`` (`float`) - :math:`\alpha` - decay rate of the cutoff
          term
          :math:`f_C(r)` (*default*: 3.0) :math:`[\mathrm{dimensionless}]`
        * ``n`` (`float`) - :math:`n` - power in
          :math:`b_{ij}` (*default*: 0.0) :math:`[\mathrm{dimensionless}]`
        * ``gamma`` (`float`) - :math:`\gamma` - coefficient in
          :math:`b_{ij}` (*default*: 0.0) :math:`[\mathrm{dimensionless}]`
        * ``c`` (`float`) - :math:`c` - coefficient in
          :math:`g(\theta)` (*default*: 0.0) :math:`[\mathrm{dimensionless}]`
        * ``d`` (`float`) - :math:`d` - coefficient in
          :math:`g(\theta)` (*default*: 1.0) :math:`[\mathrm{dimensionless}]`
        * ``m`` (`float`) - :math:`m` - coefficient in
          :math:`g(\theta)` (*default*: 0.0) :math:`[\mathrm{dimensionless}]`

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]

    Example::

        nl = md.nlist.Cell()
        tersoff = md.many_body.Tersoff(default_r_cut=1.3, nlist=nl)
        tersoff.params[('A', 'B')] = dict(magnitudes=(2.0, 1.0), lambda3=5.0)
    """
    _cpp_class_name = "PotentialTersoff2"

    def __init__(self, nlist, default_r_cut=None):
        super().__init__(nlist, default_r_cut)
        params = TypeParameter(
            'params', 'particle_types',
            TypeParameterDict(cutoff_thickness=0.2,
                              magnitudes=(1.0, 1.0),
                              exp_factors=(2.0, 1.0),
                              lambda3=0.0,
                              dimer_r=1.5,
                              n=0.0,
                              gamma=0.0,
                              c=0.0,
                              d=1.0,
                              m=0.0,
                              alpha=3.0,
                              len_keys=2))
        self._add_typeparam(params)

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
                               TypeParameterDict(A=0.0, B=float, len_keys=2))
        self._add_typeparam(params)
