// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include <pybind11/pybind11.h>
#include "PotentialValency.h"
#include "PotentialValencyGPU.h"
#include "EvaluatorVBC.h"

namespace hoomd
{
namespace md
{
namespace detail
{

void export_PotentialVBC(pybind11::module& m)
{
    export_PotentialValency<EvaluatorVBC>(m, "PotentialVBC2");
}

#ifdef ENABLE_HIP
void export_PotentialVBCGPU(pybind11::module& m)
{
    export_PotentialValencyGPU<EvaluatorVBC>(m, "PotentialVBC2GPU");
}
#endif
}
}
}

namespace py = pybind11;

PYBIND11_MODULE(_vbc_plugin, m)
{
    m.doc() = "VBC plugin for HOOMD-blue";

    hoomd::md::detail::export_PotentialVBC(m);

#ifdef ENABLE_HIP
    hoomd::md::detail::export_PotentialVBCGPU(m);
#endif
}
