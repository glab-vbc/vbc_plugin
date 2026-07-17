# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""VBC valency-capped soft density pair potential - HOOMD plugin Python API."""
import copy
import warnings

import hoomd
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeconverter import positive_real
from hoomd.data.typeparam import TypeParameter
from hoomd.md import _md
from hoomd.md.force import Force
from hoomd.vbc_plugin import _vbc_plugin

# ---------------------------------------------------------------------------
# NOTE: the `Triplet` base class below is copied verbatim from HOOMD-blue
# v4.8.2 (hoomd/md/many_body.py). The only functional change is in
# `_attach_hook`, which resolves the compiled C++ class from this plugin's
# `_vbc_plugin` module instead of HOOMD's built-in `_md`. It is kept as a
# local copy rather than `from hoomd.md.many_body import Triplet` to
# insulate the plugin from changes to that semi-private HOOMD API across
# versions.
# ---------------------------------------------------------------------------
class Triplet(Force):

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
            cls = getattr(_vbc_plugin, self._cpp_class_name)
        else:
            cls = getattr(_vbc_plugin, self._cpp_class_name + "GPU")

        self._cpp_obj = cls(self._simulation.state._cpp_sys_def,
                            self.nlist._cpp_obj)


class _ValencyParams(TypeParameter):
    """`params` accessor that keeps the asymmetric ncut_ij/ncut_ji aligned with
    the alphabetically-sorted type-pair key HOOMD stores under: on write it hands
    HOOMD an already-canonical entry (sorted couple + swapped caps); on read it
    swaps the caps back so a reversed query round-trips to what the user typed."""

    @staticmethod
    def _reversed_pair(key):
        return (isinstance(key, tuple) and len(key) == 2
                and all(isinstance(k, str) for k in key)
                and key[0] > key[1])

    def __setitem__(self, key, value):
        if (self._reversed_pair(key) and isinstance(value, dict)
                and 'ncut_ij' in value and 'ncut_ji' in value):
            key = (key[1], key[0])
            value = dict(value)
            value['ncut_ij'], value['ncut_ji'] = value['ncut_ji'], value['ncut_ij']
        super().__setitem__(key, value)

    def __getitem__(self, key):
        if not self._reversed_pair(key):
            return super().__getitem__(key)
        value = super().__getitem__((key[1], key[0]))
        try:
            swap = 'ncut_ij' in value and 'ncut_ji' in value
        except TypeError:
            swap = False
        if swap:
            value = dict(value)
            value['ncut_ij'], value['ncut_ji'] = value['ncut_ji'], value['ncut_ij']
        return value


class VBC(Triplet):
    """Valency-capped soft density pair potential.

    A many-body force giving each particle a repulsive core, a short attractive
    well, and a cap on its number of attractive neighbours - valency.
    ``use_global_dens`` selects whether the cap limits the total neighbour count
    (1) or the count per neighbour type (0).

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        default_r_cut (float): Default cutoff radius.

    Per-type-pair ``params`` keys: ``E_r`` (repulsion energy), ``E_a`` (attraction
    energy), ``rep_rcut`` / ``attr_rcut`` (repulsion / attraction cutoffs),
    ``density_lower_rcut`` / ``density_upper_rcut`` (density-counting window),
    ``ncut_ij`` / ``ncut_ji`` (valency caps of the pair's two types), and
    ``use_global_dens`` (0 = per-type cap, 1 = total cap).

    Example::

        nl = hoomd.md.nlist.Cell(buffer=0.4)
        vbc = vbc_potential.VBC(nl, default_r_cut=2.0)
        vbc.params[('V1', 'V2')] = dict(E_r=20, E_a=-4, rep_rcut=1.0, attr_rcut=1.5,
                                        density_lower_rcut=1.5, density_upper_rcut=2.0,
                                        ncut_ij=1, ncut_ji=1, use_global_dens=1)
    """
    _cpp_class_name = "PotentialVBC2"

    def __init__(self, nlist, default_r_cut=None):
        super().__init__(nlist, default_r_cut)
        params = _ValencyParams('params', 'particle_types',
                                TypeParameterDict(E_r=0.0, E_a=float, attr_rcut=1.5, rep_rcut=1.0, density_upper_rcut=2.0, density_lower_rcut=1.5, ncut_ij=1.0, ncut_ji=1.0, use_global_dens=0, len_keys=2))
        self._add_typeparam(params)
