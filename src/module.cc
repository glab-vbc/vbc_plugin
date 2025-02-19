#include <pybind11/pybind11.h>
#include "PotentialValency.h"
#include "PotentialValencyGPU.h"
#include "EvaluatorSquareDensity.h"

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
void export_PotentialSquareDensity(pybind11::module& m)
{
    export_PotentialValency<EvaluatorSquareDensity>(m, "PotentialSquareDensity2");
}

#ifdef ENABLE_HIP
void export_PotentialSquareDensityGPU(pybind11::module& m)
{
    export_PotentialValencyGPU<EvaluatorSquareDensity>(m, "PotentialSquareDensity2GPU");
}
#endif
} // end namespace detail
} // end namespace md
} // end namespace hoomd

namespace py = pybind11;

PYBIND11_MODULE(_sqd_plugin, m)
{
    m.doc() = "Squared density plugin for HOOMD-blue";

    // Register Tersoff potential
    // hoomd::md::export_PotentialTersoff(m);

    // Register Square Density potential
    hoomd::md::detail::export_PotentialSquareDensity(m);

#ifdef ENABLE_HIP
    hoomd::md::detail::export_PotentialSquareDensityGPU(m);
#endif
}
