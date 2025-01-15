#include <pybind11/pybind11.h>
#include "PotentialTersoff2.h"
#include "EvaluatorSquareDensity.h"

namespace hoomd
{
namespace md
{
// namespace detail
// {

// void export_PotentialTersoff(pybind11::module& m)
// {
//     export_PotentialTersoff2<EvaluatorTersoff>(m, "PotentialTersoff2");
// }
void export_PotentialSquareDensity(pybind11::module& m)
{
    export_PotentialTersoff2<EvaluatorSquareDensity>(m, "PotentialSquareDensity2");
}
    
// } // end namespace detail
} // end namespace md
} // end namespace hoomd

namespace py = pybind11;

PYBIND11_MODULE(_sqd_plugin, m)
{
    m.doc() = "Squared density plugin for HOOMD-blue";

    // Register Tersoff potential
    // hoomd::md::export_PotentialTersoff(m);

    // Register Square Density potential
    hoomd::md::export_PotentialSquareDensity(m);
}
