// Copyright (c) 2025 Goloborodko Lab.
// Released under the BSD 3-Clause License.

/*! \file HoomdCompat.h
    \brief Shim that lets the plugin build against both HOOMD-blue 4.8.x (the
           version it was written for) and 5.x/6.x/7.x.

    `GlobalArray` was removed in HOOMD 5.0; `NeighborList::addRCutMatrix` takes
    a `GPUArray` there. `VBCRCutArray` names whichever one the install provides.
*/

#pragma once

#include "hoomd/HOOMDMath.h"

#if defined(__has_include)
#if __has_include("hoomd/GlobalArray.h")
#define VBC_HAS_GLOBAL_ARRAY
#endif
#else
// No __has_include: assume the old header exists (pre-5.x toolchains).
#define VBC_HAS_GLOBAL_ARRAY
#endif

// The array headers are host-only.
#ifndef __HIPCC__

#ifdef VBC_HAS_GLOBAL_ARRAY
#include "hoomd/GlobalArray.h"
#else
#include "hoomd/GPUArray.h"
#endif

namespace hoomd
    {
namespace md
    {
#ifdef VBC_HAS_GLOBAL_ARRAY
template<class T> using VBCRCutArray = GlobalArray<T>;
#else
template<class T> using VBCRCutArray = GPUArray<T>;
#endif
    } // end namespace md
    } // end namespace hoomd

#endif // __HIPCC__
