#ifndef ALGOIM_REPARAMETERIZATION_BSPLINE_H
#define ALGOIM_REPARAMETERIZATION_BSPLINE_H

/* Creates a reparameterization of a tensor-product BSpline function. */

#include "real.hpp"
#include "uvector.hpp"
#include "hyperrectangle.hpp"
#include "bspline.hpp"
#include "lagrange.hpp"
#include "bezier.hpp"
#include "reparameterization_bezier.hpp"
#include "quadrature_general_decomposition.hpp"
#include "reparameterization_common.hpp"

#include <cmath>
#include <memory>

namespace algoim::bspline
{
namespace detail
{

    /**
     * @brief Appends the given reparameterization cells to a new vector after
     * transforming their image from the unit domain [0,1]^N to the given one.
     * 
     * @tparam N Parametric dimension of the given reparameterization cells.
     * @tparam ElemPtr Type of the reparameterization cells (shared pointer).
     * @param _to_append Vector of reparameterizations to (transform and) append.
     * @param _domain Domain to which the reparameterization cells are transformed
     *        from the [0,1]^N  domain.
     * @param _reps Vector where the reparameterization are appended.
     * 
     * @note This function is thread-safe for OpenMP.
     */
    template<int N, typename ElemPtr>
    void appendReparams(const std::vector<ElemPtr> &_to_append,
                        const HyperRectangle<real, N> &_domain,
                        std::vector<ElemPtr> &_reps)
    {
        const HyperRectangle<real, N> unit_domain(0.0, 1.0);
        for(const auto &rep : _to_append)
        {
            rep->transformImage(unit_domain, _domain);
          	#pragma omp critical
            {
                _reps.push_back(rep);
            }
        }
    };

    /**
     * @brief Reparameterize the full domain of given Bsplines.
     * 
     * @tparam N Parametric dimension of the given BSplines.
     * 
     * @param _splines Vector of Bspline to reparameterize.
     * @param _order Order of the reparameterization (number of points
     *        per direction in each reparameterization cell).
     * @param _split_in_Beziers Flag indicating whether the generated
     *        reparameterization must be conformal with the Bezier elements
     *        of the Bspline functions (true) or not (false).
     * @param _reps Vector where the reparameterization are appended.
     */
    template<int N>
    void reparamFull(const std::vector<std::shared_ptr<const BSplineTP<N, 1>>> &_splines,
                     const int _order,
                     const bool _split_in_Beziers,
                     std::vector<std::shared_ptr<algoim::detail::ReparamElemType<N,N>>> &_reps)
    {
        using Elem = algoim::detail::ReparamElemType<N,N>;

        const HyperRectangle<real, N> unit_domain(0, 1);

        const auto n = static_cast<int>(_splines.size());
        const auto unit_reparam = std::make_shared<Elem>(unit_domain, _order);

        #pragma omp parallel for
        for(int i = 0; i < n; ++i)
        {
            const auto spline = _splines[i];

            if (_split_in_Beziers)
            {
                for(const auto &domain : spline->getElementsDomains())
                {
                    const auto reparam = unit_reparam->clone();
                    appendReparams<N>({reparam}, domain, _reps);
                }
            }
            else
            {
                const auto domain = spline->getDomain();
                const auto reparam = unit_reparam->clone();
                appendReparams<N>({reparam}, domain, _reps);
            }
        } // i
    }

    /**
     * @brief Reparameterize the full domain of given Bsplines creating
     * only the edges wirebasket.
     * 
     * @tparam N Parametric dimension of the given BSplines.
     * 
     * @param _splines Vector of Bspline to reparameterize.
     * @param _order Order of the reparameterization (number of points
     *        per direction in each reparameterization cell).
     * @param _split_in_Beziers Flag indicating whether the generated
     *        reparameterization must be conformal with the Bezier elements
     *        of the Bspline functions (true) or not (false).
     * @param _reps Vector where the reparameterization are appended.
     */
    template<int N>
    void reparamFullWirebasket(const std::vector<std::shared_ptr<const BSplineTP<N, 1>>> &_splines,
                                 const int _order,
                                 const bool _split_in_Beziers,
                                 std::vector<std::shared_ptr<algoim::detail::ReparamElemType<1,N>>> &_reps)
    {
        const auto get_unit_reparams = [_order]()
        {
            static thread_local const auto unit_reparam = algoim::detail::createHypercubeWirebasket<N>(_order);
            std::vector<std::shared_ptr<algoim::detail::ReparamElemType<1,N>>> reps;
            reps.reserve(unit_reparam.size());
            for(const auto &rep : unit_reparam)
                reps.push_back(rep->clone());
            return reps;
        };

        const auto n = static_cast<int>(_splines.size());

        #pragma omp parallel for
        for(int i = 0; i < n; ++i)
        {
            const auto spline = _splines[i];

            if (_split_in_Beziers)
            {
                for(const auto &domain : spline->getElementsDomains())
                    appendReparams<N>(get_unit_reparams(), domain, _reps);
            }
            else
            {
                const auto domain = spline->getDomain();
                appendReparams<N>(get_unit_reparams(), domain, _reps);
            }
        } // i
    }

} // detail

/**
 * @brief Reparameterizes a Bspline implicit function.
 * 
 * @tparam N Parametric dimension of the function.
 * @tparam S Flag indicating if the reparameterization must
 *         be performed only for the levelset surface (true), i.e.,
 *         the manifold where the Bspline function is equal to 0,
 *         or the volume (false), i.e., the subregion where
 *         the Bspline function is negative.
 * @param _spline Bspline implicit function to reparameterize.
 * @param _order Order of the reparameterization (number of points
 *        per direction in each reparameterization cell).
 * @param _split_in_Beziers Flag indicating whether the generated
 *        reparameterization must be conformal with the Bezier elements
 *        of the Bspline function (true) or not (false).
 * @return Vector of Lagrange elements that reparameterize
 *        either the surface or the volume.
 */
template<int N, bool S = false>
std::vector<std::shared_ptr<algoim::detail::ReparamElemType<S?N-1:N,N>>>
reparam(const BSplineTP<N, 1> &_spline,
        const int _order,
        const bool _split_in_Beziers)
{
    using SplinePtr = std::shared_ptr<const BSplineTP<N, 1>>;
    std::vector<SplinePtr> internals, externals, unknowns;
    bspline::decompose(_spline, internals, externals, unknowns);

    using Elem = algoim::detail::ReparamElemType<S?N-1:N,N>;
    using ElemPtr = std::shared_ptr<Elem>;

    std::vector<ElemPtr> reps;
    reps.reserve((S ? 0 : internals.size()) + unknowns.size() * 6);

    if constexpr (!S)
    {
        detail::reparamFull<N>(internals, _order, _split_in_Beziers, reps);
    }

    const auto n_unknowns = static_cast<int>(unknowns.size());

    #pragma omp parallel for
    for(int i = 0; i < n_unknowns; ++i)
    {
        const auto &unknown = unknowns[i];

        const auto bzr = unknown->transformToBezier();
        auto reparams = bezier::reparam<N, S>(bzr->getXarray(), _order);

        const auto domain = unknown->getDomain();
        detail::appendReparams<N>(reparams, domain, reps);
    }

    return reps;
}

/**
 * @brief Reparameterizes a domain defined by two Bspline implicit functions.
 * Both Bspline functions must have the same parametric space.
 * 
 * @tparam N Parametric dimension of the function.
 * @tparam S Flag indicating if the reparameterization must
 *         be performed only for the levelset surface (true), i.e.,
 *         the manifold where the either of the two Bezier functions are
 *         equal to 0, or the subregion (false) between those surfaces,
 *         where both Bspline functions are negative.
 * @param _spline0 First Bspline implicit function defining the domain.
 * @param _spline1 Second Bspline implicit function defining the domain.
 * @param _order Order of the reparameterization (number of points
 *        per direction in each reparameterization cell).
 * @param _split_in_Beziers Flag indicating whether the generated
 *        reparameterization must be conformal with the Bezier elements
 *        of the Bspline function (true) or not (false).
 * @return Vector of Lagrange elements that reparameterize
 *        either the surface or the volume.
 */
template<int N, bool S = false>
std::vector<std::shared_ptr<algoim::detail::ReparamElemType<S?N-1:N,N>>>
reparam(const BSplineTP<N, 1> &_spline0,
        const BSplineTP<N, 1> &_spline1,
        const int _order,
        const bool _split_in_Beziers)
{
    using SplinePtr = std::shared_ptr<const BSplineTP<N, 1>>;
    std::vector<std::array<SplinePtr, 2>> internals, externals, unknowns;
    bspline::decompose(_spline0, _spline1, internals, externals, unknowns);

    using Elem = algoim::detail::ReparamElemType<S?N-1:N,N>;
    using ElemPtr = std::shared_ptr<Elem>;

    std::vector<std::shared_ptr<Elem>> reps;
    reps.reserve((S ? 0 : internals.size()) + unknowns.size() * 6);

    if constexpr (!S)
    {
        std::vector<SplinePtr> internals_0;
        internals_0.reserve(internals.size());
        for(const auto &internal : internals)
            internals_0.push_back(internal[0]);

        detail::reparamFull<N>(internals_0, _order, _split_in_Beziers, reps);
    }

    const auto n_unknowns = static_cast<int>(unknowns.size());
    for(int i = 0; i < n_unknowns; ++i)
    {
        const auto &unknown = unknowns[i];

        const auto bzr0 = unknown[0]->transformToBezier();
        const auto bzr1 = unknown[1]->transformToBezier();
        const auto domain = unknown[0]->getDomain();

        auto reparams = bezier::reparam<N, S>(bzr0->getXarray(), bzr1->getXarray(), _order);

        detail::appendReparams<N>(reparams, domain, reps);
    }

    return reps;
}

/**
 * @brief Reparameterizes a Bspline implicit function creating only the edges wirebasket.
 * 
 * @tparam N Parametric dimension of the function.
 * @tparam S Flag indicating if the reparameterization must
 *         be performed only for the levelset surface (true), i.e.,
 *         the manifold where the Bspline function is equal to 0,
 *         or the volume (false), i.e., the subregion where
 *         the Bspline function is negative.
 * @param _spline Bspline implicit function to reparameterize.
 * @param _order Order of the reparameterization (number of points
 *        per direction in each reparameterization cell).
 * @param _split_in_Beziers Flag indicating whether the generated
 *        reparameterization must be conformal with the Bezier elements
 *        of the Bspline function (true) or not (false).
 * @return Vector of Lagrange elements that reparameterize
 *        either the surface or the volume.
 */
template<int N, bool S = false>
std::vector<std::shared_ptr<algoim::detail::ReparamElemType<1,N>>>
reparamWirebasket(const BSplineTP<N, 1> &_spline,
                   const int _order,
                   const bool _split_in_Beziers)
{
    using SplinePtr = std::shared_ptr<const BSplineTP<N, 1>>;
    std::vector<SplinePtr> internals, externals, unknowns;
    bspline::decompose(_spline, internals, externals, unknowns);

    using Elem = algoim::detail::ReparamElemType<1,N>;
    using ElemPtr = std::shared_ptr<Elem>;

    std::vector<std::shared_ptr<Elem>> reps;
    reps.reserve(((S ? 0 : internals.size()) + unknowns.size() * 6) * 12);

    if constexpr (!S)
    {
        detail::reparamFullWirebasket<N>(internals, _order, _split_in_Beziers, reps);
    }

    const auto n_unknowns = static_cast<int>(unknowns.size());
    for(int i = 0; i < n_unknowns; ++i)
    {
        const auto &unknown = unknowns[i];

        const auto bzr = unknown->transformToBezier();
        auto reparams = bezier::reparamWirebasket<N, S>(bzr->getXarray(), _order);

        const auto domain = unknown->getDomain();
        detail::appendReparams<N>(reparams, domain, reps);
    }

    return reps;
}

/**
 * @brief Reparameterizes a domain defined by two Bspline implicit functions creating only the edges wirebasket.
 * Both Bspline functions must have the same parametric space.
 * 
 * @tparam N Parametric dimension of the function.
 * @tparam S Flag indicating if the reparameterization must
 *         be performed only for the levelset surface (true), i.e.,
 *         the manifold where the either of the two Bezier functions are
 *         equal to 0, or the subregion (false) between those surfaces,
 *         where both Bspline functions are negative.
 * @param _spline0 First Bspline implicit function defining the domain.
 * @param _spline1 Second Bspline implicit function defining the domain.
 * @param _order Order of the reparameterization (number of points
 *        per direction in each reparameterization cell).
 * @param _split_in_Beziers Flag indicating whether the generated
 *        reparameterization must be conformal with the Bezier elements
 *        of the Bspline function (true) or not (false).
 * @return Vector of Lagrange elements that reparameterize
 *        either the surface or the volume.
 */
template<int N, bool S = false>
std::vector<std::shared_ptr<algoim::detail::ReparamElemType<1,N>>>
reparamWirebasket(const BSplineTP<N, 1> &_spline0,
                   const BSplineTP<N, 1> &_spline1,
                   const int _order,
                   const bool _split_in_Beziers)
{
    using SplinePtr = std::shared_ptr<const BSplineTP<N, 1>>;
    std::vector<std::array<SplinePtr, 2>> internals, externals, unknowns;
    bspline::decompose(_spline0, _spline1, internals, externals, unknowns);

    using Elem = algoim::detail::ReparamElemType<1,N>;
    using ElemPtr = std::shared_ptr<Elem>;

    std::vector<std::shared_ptr<Elem>> reps;
    reps.reserve(((S ? 0 : internals.size()) + unknowns.size() * 6) * 12);

    if constexpr (!S)
    {
        std::vector<SplinePtr> internals_0;
        internals_0.reserve(internals.size());
        for(const auto &internal : internals)
            internals_0.push_back(internal[0]);

        detail::reparamFullWirebasket<N>(internals_0, _order, _split_in_Beziers, reps);
    }

    const auto n_unknowns = static_cast<int>(unknowns.size());
    for(int i = 0; i < n_unknowns; ++i)
    {
        const auto &unknown = unknowns[i];

        const auto bzr0 = unknown[0]->transformToBezier();
        const auto bzr1 = unknown[1]->transformToBezier();
        const auto domain = unknown[0]->getDomain();

        auto reparams = bezier::reparamWirebasket<N, S>(bzr0->getXarray(), bzr1->getXarray(), _order);

        detail::appendReparams<N>(reparams, domain, reps);
    }

    return reps;
}

} // namespace algoim::bspline

#endif // ALGOIM_REPARAMETERIZATION_BSPLINE_H