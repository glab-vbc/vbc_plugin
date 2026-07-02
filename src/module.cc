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

// void export_PotentialTersoff(pybind11::module& m)
// {
//     export_PotentialValency<EvaluatorTersoff>(m, "PotentialValency");
// }
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
} // end namespace detail
} // end namespace md
} // end namespace hoomd

namespace py = pybind11;

PYBIND11_MODULE(_vbc_plugin, m)
{
    m.doc() = "VBC plugin for HOOMD-blue";

    // Register Tersoff potential
    // hoomd::md::export_PotentialTersoff(m);

    // Register VBC potential
    hoomd::md::detail::export_PotentialVBC(m);

#ifdef ENABLE_HIP
    hoomd::md::detail::export_PotentialVBCGPU(m);
#endif
}
