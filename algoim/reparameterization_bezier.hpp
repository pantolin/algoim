#ifndef ALGOIM_REPARAMETERIZATION_BEZIER_H
#define ALGOIM_REPARAMETERIZATION_BEZIER_H

/* Creates a reparameterization of a tensor-product Bezier functions. */

#include "reparameterization_common.hpp"

#include "quadrature_multipoly.hpp"
#include "bernstein.hpp"
#include "hyperrectangle.hpp"
#include "lagrange.hpp"
#include "real.hpp"
#include "uvector.hpp"
#include "xarray.hpp"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>

/* High-order accurate reparameterization algorithms for implicitly defined domains through Bezier polynomials
   in the unit hypercube, based on the quadrature algorithms implemented in ImplicitPolyQuadrature and described in the paper:
    - R. I. Saye, High-order quadrature on multi-component domains implicitly defined by multivariate polynomials,
      Journal of Computational Physics, 448, 110720 (2022).
      https://doi.org/10.1016/j.jcp.2021.110720 */

namespace algoim::bezier
{
namespace detail
{

    /**
     * @brief M-dimensional reparameterization of an N-dimensional Bezier(s) polynomial(s)
     * in the unit hypercube.
     * This is just a modification of the class ImplicitPolyQuadrature (quadrature_multipoly.hpp)
     * for generating reparameterizations.
     * 
     * @tparam M Parametric dimension of the current sub-domain.
     * @tparam N Parametric dimension of the final domain.
     * @tparam S Flag indicating if the reparameterization must
     *         be performed only for the levelset surface (true), i.e.,
     *         the manifold where the any of the polynomials is equal to 0,
     *         or the subregion volume (false) between those manifolds
     *         where all the polynomials are negative.
     */
    template<int M, int N, bool S>
    struct ImplicitPolyReparam
    {
        /// Integral type.
        enum IntegralType { Inner, OuterSingle, OuterAggregate };

        /// Type of the parent, either a higher-dimension class instance or void (when M==N).
        using Parent = std::conditional_t<M==N,void, ImplicitPolyReparam<M+1, N, S>>;

        /// Tolerance to be used in certain calculations.
        static constexpr real tol = 10.0 * std::numeric_limits<real>::epsilon();

        /// Given M-dimensional polynomials
        PolySet<M,ALGOIM_M> phi;
        /// Elimination axis/height direction; k=M if there are no interfaces, k=-1 if the domain is empty.
        int k;
        /// Reparameterization order.
        int order;
        /// Base polynomials corresponding to removal of axis k
        ImplicitPolyReparam<M-1,N, S> base;
        /// If quad method is auto chosen, indicates whether TS is applied
        bool auto_apply_TS;
        /// Whether an inner integral, or outer of two kinds
        IntegralType type;
        /// Stores other base cases, besides k, when in aggregate mode.
        std::array<std::tuple<int,ImplicitPolyReparam<M-1, N, S>>,M-1> base_other;
        /// Parent class instance (or void when M==N).
        Parent *parent;
        /// Reference intervals in the unit hypercube along the current height direction.
        std::map<uvector<int,M-1>, algoim::detail::RootsIntervals<M-1>, util::UvectorCompare<int, M-1>> ref_intervals;
        /// Reparameterization element (if M < N, void).
        std::conditional_t<M == N, algoim::detail::ReparamElems<S?N-1:N, N>, void *> reparam_elems;

        /// Default constructor sets to an uninitialised state.
        ImplicitPolyReparam() : k(-1)
        {
            this->setOrder(2);
        }

        /**
         * @briefs Build quadrature hierarchy for a domain implicitly defined by a single polynomial
         * 
         * @param _p Implicit polynomial.
         * @param _order Reparameterization order.
         */
        ImplicitPolyReparam(const xarray<real,M>& _p, const int _order)
        {
            this->appendPolynomial(_p);
            this->setOrder(_order);
            this->build(true, false);
        }

        /**
         * @brief Builds quadrature hierarchy for a domain implicitly defined by two polynomials
         * 
         * @param _p First implicit polynomial.
         * @param _q First implicit polynomial.
         * @param _order Reparameterization order.
         */
        ImplicitPolyReparam(const xarray<real,M>& _p, const xarray<real,M>& _q, const int _order)
        {
            this->appendPolynomial(_p);
            this->appendPolynomial(_q);
            this->setOrder(_order);
            this->build(true, false);
        }

        /**
         * @brief Adds a new implicit polynomial.
         * 
         * @param _p Implicit polynomial to append.
         * @param _mask Mask of the polynomial.
         */
        void appendPolynomial(const xarray<real,M>& _p,
                              const booluarray<M,ALGOIM_M>& _pmask = booluarray<M,ALGOIM_M>(true))
        {
            auto mask = algoim::detail::nonzeroMask(_p, _pmask);
            if (!algoim::detail::maskEmpty(mask))
                phi.push_back(_p, mask);
        }

        // Build quadrature hierarchy for a given domain implicitly defined by two polynomials with user-defined masks

        /**
         * @brief Builds quadrature hierarchy for a given domain implicitly defined by two polynomials with user-defined masks
         * 
         * @param _p First implicit polynomial.
         * @param _pmask Mask of the first polynomial.
         * @param _q First implicit polynomial.
         * @param _qmask Mask of the second polynomial.
         * @param _order Reparameterization order.
         */
        ImplicitPolyReparam(const xarray<real,M>& _p, 
                            const booluarray<M,ALGOIM_M>& _pmask,
                            const xarray<real,M>& _q,
                            const booluarray<M,ALGOIM_M>& _qmask,
                            const int _order)
        {
            this->appendPolynomial(_p, _pmask);
            this->appendPolynomial(_q, _qmask);
            this->setOrder(_order);
            this->build(true, false);
        }

        /**
         * @brief Sets the reparameterization order.
         * @param _order Reparameterization order.
         */
        void setOrder(const int _order)
        {
            assert(1 < _order);

            this->order = _order;
            if constexpr (M == N)
                reparam_elems.p = _order;
        }

        /**
         * @brief Assuming phi has been instantiated, determine elimination axis and build base
         * 
         * @param outer 
         * @param auto_apply_TS 
         */
        void build(bool outer, bool auto_apply_TS)
        {
            type = outer ? OuterSingle : Inner;
            this->auto_apply_TS = auto_apply_TS;

            // If phi is empty, apply a tensor-product Gaussian quadrature
            if (phi.count() == 0)
            {
                k = M;
                this->auto_apply_TS = false;
                return;
            }

            if constexpr (M == 1)
            {
                // If in one dimension, there is only one choice of height direction and
                // the recursive process halts
                k = 0;
                return;
            }
            else
            {
                // Compute score; penalise any directions which likely contain vertical tangents
                uvector<bool,M> has_disc;
                uvector<real,M> score = algoim::detail::score_estimate(phi, has_disc);
                assert(max(abs(score)) > 0);
                score /= 2 * max(abs(score));
                for (int i = 0; i < M; ++i)
                    if (!has_disc(i))
                        score(i) += 1.0;

                // Choose height direction and form base polynomials; if tanh-sinh is being used at this
                // level, suggest the same all the way down; moreover, suggest tanh-sinh if a non-empty
                // discriminant mask has been found
                k = argmax(score);
                algoim::detail::eliminate_axis(phi, k, base.phi);
                base.parent = this;
                base.order = this->order;
                base.build(false, this->auto_apply_TS || has_disc(k));

                // If this is the outer integral, and surface quadrature schemes are required, apply
                // the dimension-aggregated scheme when necessary
                if (outer && has_disc(k))
                {
                    type = OuterAggregate;
                    for (int i = 0; i < M; ++i) if (i != k)
                    {
                        auto& [kother, base] = base_other[i < k ? i : i - 1];
                        kother = i;
                        algoim::detail::eliminate_axis(phi, kother, base.phi);
                        // In aggregate mode, triggered by non-empty discriminant mask,
                        // base integrals always have T-S suggested
                        base.parent = this;
                        base.order = this->order;
                        base.build(false, this->auto_apply_TS || true);
                    }
                }
            }
        }


        /**
         * @brief Clears recursively the reference intervals and associated points.
         * It clears the intervals for the current dimension and higher ones.
         */
        void clearReferenceIntervals()
        {
            this->ref_intervals.clear();
            if constexpr (M < N)
                this->parent->clearReferenceIntervals();
        }

        /**
         * @brief Computes the roots of the polynomial @p _poly_id at point @p _x, 
         * along the direction k.
         * 
         * @param _x Point at which the roots are computed.
         * @param _poly_id Id of the polynomial.
         * @return Computed roots.
         */
        std::vector<real> computeRoots(uvector<real,M-1> _x, const int _poly_id)
        {
            std::vector<real> roots;

            const auto& p = this->phi.poly(_poly_id);
            const auto& mask = this->phi.mask(_poly_id);
            int P = p.ext(this->k);

            // Ignore phi if its mask is void everywhere above the base point
            if (!algoim::detail::lineIntersectsMask(mask, _x, this->k))
                return roots;

            // Restrict polynomial to axis-aligned line and compute its roots
            real *pline, *roots_i;
            algoim_spark_alloc(real, &pline, P, &roots_i, P - 1);
            bernstein::collapseAlongAxis(p, _x, this->k, pline);
            const int rcount = algoim::bernstein::bernsteinUnitIntervalRealRoots(pline, P, roots_i);

            // Add all real roots in [0,1] which are also within masked region of phi
            for (int j = 0; j < rcount; ++j)
            {
                const auto x = add_component(_x, this->k, roots_i[j]);
                if (algoim::detail::pointWithinMask(mask, x))
                    roots.push_back(roots_i[j]);
            }

            return roots;
        }

        /**
         * @brief Checks if a point is inside the implicit domain described by polynomials.
         * We consider that the point is inside if at that point all the polynomials are negative.
         * 
         * @param _x Point to be checked.
         * @return True if points inside, false otherwise.
         */
        bool checkPointInside(const uvector<real,M> &_x)
        {
            for(int i = 0; i < this->phi.count(); ++i)
            {
                const auto poly = this->phi.poly(i);
                if (0.0 < bernstein::evalBernsteinPoly(poly, _x))
                    return false;
            }
            return true;
        }

        /**
         * @brief Computes all the intervals (between consecutive roots)
         * at point @p _x along direction k.
         * 
         * @param _x Point at which intervals are computed.
         * @return Computed intervals.
         */
        algoim::detail::RootsIntervals<M-1> computeAllIntervals(uvector<real,M-1> _x)
        {
            algoim::detail::RootsIntervals<M-1> intervals;
            intervals.point = _x;
            auto &roots = intervals.roots;
            auto &func_ids = intervals.func_ids;

            intervals.addRoot(0.0, -1);
            intervals.addRoot(1.0, -1);

            for (int i = 0; i < this->phi.count(); ++i)
            {
                for(const auto r : this->computeRoots(_x, i))
                    intervals.addRoot(r, i);
            }

            // In rare cases, degenerate segments can be found, filter out with a tolerance
            intervals.adjustRoots(tol, 0.0, 1.0);

            const auto n_int = intervals.getNumRoots() - 1;
            assert(0 < n_int);

            intervals.active_intervals.resize(n_int);

            for(int i = 0; i < n_int; ++i)
            {
                const real x0 = intervals.roots[i];
                const real x1 = intervals.roots[i+1];
                intervals.active_intervals[i] = std::abs(x1 - x0) > tol;
            }

            return intervals;
        }

        /**
         * @brief Checks if the intervals with the given @p _elem_tid_base,
         * and the ones higher dimension and same base @p _elem_tid_base, are inactive.
         * 
         * @param _elem_tid_base Element tensor-index of the intervals to be queried.
         * @return True if all the intervals are inactive, false otherwise.
         */
        bool checkAllReferenceIntervalsInactive(const uvector<int,M-1> _elem_tid_base) const
        {
            const auto &intervals = this->ref_intervals.at(_elem_tid_base);
            const auto &active_intervals = intervals.active_intervals;

            const auto n_int = static_cast<int>(active_intervals.size());
            for(int i = 0; i < n_int; ++i)
            {
                if (active_intervals[i])
                {
                    if constexpr (M < N)
                    {
                        const auto elem_tid = add_component(_elem_tid_base, this->k, i);
                        if (!this->parent->checkAllReferenceIntervalsInactive(elem_tid))
                            return false;
                    }
                    else // (M == N)
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        /**
         * @brief Computes recursively the reference intervals for the current
         * dimension and the ones above.
         * 
         * @param _x_base Point at which the intervals are computed.
         * @param _elem_tid_base Element tensor index.
         */
        void computeReferenceIntervals(uvector<real,M-1> _x_base, uvector<int,M-1> _elem_tid_base)
        {
            if constexpr (M == 1)
            {
                this->parent->clearReferenceIntervals();
            }

            auto intervals = this->computeAllIntervals(_x_base);

            // Loop over segments of divided interval
            const auto n_int = static_cast<int>(intervals.active_intervals.size());
            for(int i = 0; i < n_int; ++i)
            {
                if (!intervals.active_intervals[i])
                    continue;

                const real x0 = intervals.roots[i];
                const real x1 = intervals.roots[i+1];

                const auto x = add_component(_x_base, this->k, 0.5 * (x0 + x1));
                const auto elem_tid = add_component(_elem_tid_base, this->k, i);

                if constexpr (M < N)
                {
                    this->parent->computeReferenceIntervals(x, elem_tid);
                    intervals.active_intervals[i] = !this->parent->checkAllReferenceIntervalsInactive(elem_tid);
                }
                else // (M == N)
                {
                    intervals.active_intervals[i] = this->checkPointInside(x);
                }
            } // i

            if constexpr (M == N && S)
            {
                std::vector<bool> new_active_intervals(n_int, false);
                for(int i = 0; i < (n_int-1); ++i)
                    new_active_intervals[i] = intervals.active_intervals[i] != intervals.active_intervals[i+1];
                intervals.active_intervals = new_active_intervals;
            }

            this->ref_intervals.emplace(_elem_tid_base, intervals);
        }

        /**
         * @brief Computes all the intervals (between consecutive roots)
         * at point @p _x along direction k, taking as reference the provided
         * @p _ref_intervals.
         * 
         * This method allows to deal with degenerate intervals by comparing
         * the computed ones with the ones obtained at a point without degeneracies.
         * 
         * @param _ref_intervals Reference intervals.
         * @param _x Point at which new intervals are computed.
         * @return Computed intervals.
         * 
         * @warning This method is not bulletproof, and it may fail in corner cases.
         */
        algoim::detail::RootsIntervals<M-1>
        computeSimilarIntervals(const algoim::detail::RootsIntervals<M-1> &_ref_intervals,
                                  uvector<real,M-1> _x)
        {
            static_assert(1 < M, "Invalid dimension.");

            algoim::detail::RootsIntervals<M-1> intervals;
            intervals.point = _x;

            const auto n_roots = _ref_intervals.getNumRoots();
            if (n_roots == 2)
            {
                return _ref_intervals;
            }

            const real x0{0.0};
            const real x1{1.0};

            for(int i = 0; i < this->phi.count(); ++i)
            {
                std::vector<real> roots_i;

                for (int j = 0; j < n_roots; ++j)
                {
                    if (i == _ref_intervals.func_ids[j])
                    {
                        roots_i.push_back(_ref_intervals.roots[j]);
                    }
                }

                if (roots_i.empty())
                    continue;

                const auto n_i = roots_i.size();

                auto new_roots_i = this->computeRoots(_x, i);

                // Filtering out roots near x0 and x1.
                const auto it = std::remove_if(new_roots_i.begin(), new_roots_i.end(),
                [x0,x1](const auto &_r)
                {
                    return std::abs(_r - x0) < tol || std::fabs(_r - x1) < tol;
                });
                new_roots_i.erase(it, new_roots_i.end());
                const auto n_i_new = new_roots_i.size();

                if (n_i_new < n_i)
                {
                    // We have to add roots in x0 and/or x1.
                    const auto& poly = this->phi.poly(i);
                    const bool root_0 = std::abs(bernstein::evalBernsteinPoly(poly, add_component(_x, this->k, x0))) < tol;
                    const bool root_1 = std::abs(bernstein::evalBernsteinPoly(poly, add_component(_x, this->k, x1))) < tol;

                    if (root_0 ^ root_1)
                    {
                        new_roots_i.insert(new_roots_i.end(), n_i - n_i_new, root_0 ? x0 : x1);
                    }
                    else if (root_0 && root_1)
                    {
                        if ((n_i_new + 2) == n_i)
                        {
                            new_roots_i.push_back(x0);
                            new_roots_i.push_back(x1);
                        }
                        else if ((n_i_new + 1) == n_i)
                        {
                            // We decide if inserting x0 and x1 based on the function signs.
                            const auto ref_intervals_pt = _ref_intervals.point;
                            auto x = add_component(ref_intervals_pt, this->k, 0.5 * (x0 + roots_i.front()));
                            const auto sign = bernstein::evalBernsteinPoly(poly, x) > 0;

                            std::sort(new_roots_i.begin(), new_roots_i.end());
                            const real xmid = new_roots_i.empty() ? x1 : new_roots_i.front();
                            x = add_component(_x, this->k, 0.5 * (x0 + xmid));
                            const auto new_sign = bernstein::evalBernsteinPoly(poly, x) > 0;

                            new_roots_i.push_back(sign == new_sign ? x1 : x0);
                        }
                    }
                }

                if (new_roots_i.size() != n_i) // First backup strategy.
                {
                    constexpr int n_pts = 6;
                    const real t0 = std::log(std::numeric_limits<real>::epsilon() * 10.0);
                    const real t1 = std::log(0.001);
                    const real dt = (t1 - t0) / real(n_pts-1);

                    std::vector<real> ts;
                    ts.reserve(n_pts + 3);
                    for(int j = 0; j < n_pts; ++j)
                        ts.push_back(std::exp(t0 + j * dt));
                    ts.push_back(0.005);
                    ts.push_back(0.01);
                    ts.push_back(0.05);
                    ts.push_back(0.1);
                    ts.push_back(0.5);

                    const uvector<real, M-1> &old_x = _ref_intervals.point;
                    uvector<real,M-1> new_x = _x;
                    const int dir = std::max(this->base.k, M-2);
                    for (const auto &t : ts)
                    {
                        new_x(dir) = (1.0 - t) * _x(dir) + t * old_x(dir);
                        new_roots_i = this->computeRoots(new_x, i);
                        if (new_roots_i.size() == n_i)
                            break;
                    }
                }

                if (new_roots_i.size() != n_i) // Last backup strategy.
                {
                    new_roots_i = roots_i;
                }

                for(const auto r : new_roots_i)
                    intervals.addRoot(r, i);

            } // i

            intervals.addRoot(x0, -1);
            intervals.addRoot(x1, -1);
            intervals.active_intervals = _ref_intervals.active_intervals;
            assert(intervals.getNumRoots() == _ref_intervals.getNumRoots());

            intervals.adjustRoots(tol, x0, x1);

            return intervals;
        }

        /**
         * @brief Computes all the intervals (between consecutive roots)
         * at point @p _x along direction e0.
         * In the 1D, it just computes the intervals without any reference.
         * For higher dimensions, the stored reference intervals are used.
         * 
         * @param _x Point at which new intervals are computed.
         * @param _elem_tid Element tensor index (for selecting the reference intervals).
         * @return Computed intervals.
         */
        algoim::detail::RootsIntervals<M-1>
        computeIntervals(uvector<real,M-1> _x, uvector<int,M-1> _elem_id)
        {
            const auto &ref_int = this->ref_intervals.at(_elem_id);
            if constexpr (M == 1)
                return ref_int;
            else // if (1 < M)
                return computeSimilarIntervals(ref_int, _x);
        }

        /**
         * @brief Generates the reparameterization point by point in a dimensional
         * recursive way.
         * 
         * @param _x_base Point of lower dimension.
         * @param _elem_id_base Lower dimension element tensor index.
         * @param _pt_id_base Lower dimension point tensor index.
         */
        void process(const uvector<real,M-1> &_x_base,
                     const uvector<int,M-1> &_elem_id_base,
                     const uvector<int,M-1> &_pt_id_base)
        {
            const auto intervals = computeIntervals(_x_base, _elem_id_base);

            // Loop over segments of divided interval
            const auto n_int = static_cast<int>(intervals.active_intervals.size());
            for(int i = 0; i < n_int; ++i)
            {
                if (!intervals.active_intervals[i])
                    continue;

                const real x0 = intervals.roots[i];
                const real x1 = intervals.roots[i+1];

                const auto elem_id = add_component(_elem_id_base, k, i);

                if constexpr (M == N && S)
                {
                    const auto x = add_component(_x_base, k, x1);
                    this->reparam_elems.process(x, elem_id, _pt_id_base);
                }
                else
                {
                    for (int j = 0; j < this->order; ++j)
                    {
                        const auto coord = x0 + (x1 - x0) * bernstein::modifiedChebyshevNode(j, this->order);
                        const auto x = add_component(_x_base, k, coord);

                        const auto pt_id = add_component(_pt_id_base, k, j);
                        if constexpr (M < N)
                            this->parent->process(x, elem_id, pt_id);
                        else
                            this->reparam_elems.process(x, elem_id, pt_id);
                    }
                }
            } // i
        }

        /**
         * @brief Generates the reparameterization for a (sub-dimensional) domain
         * that is not intersected by the polynomials.
         */
        void reparamTensorProduct()
        {
            if constexpr (M == N)
            {
                if constexpr (!S)
                {
                    if (this->checkPointInside(0.5))
                    {
                        using Elem = algoim::detail::ReparamElemType<N,N>;
                        const HyperRectangle<real, N> unit_domain(0.0, 1.0);
                        const auto elem = std::make_shared<Elem>(unit_domain, this->order);
                        reparam_elems.reparam_elems.push_back(elem);
                    }
                }
            }
            else // if constexpr (M < N)
            {
                parent->computeReferenceIntervals(0.5, 0);

                for (MultiLoop<M> i(0, this->order); ~i; ++i)
                {
                    uvector<real,M> x;
                    for (int dim = 0; dim < M; ++dim)
                        x(dim) = bernstein::modifiedChebyshevNode(i(dim), this->order);

                    this->parent->process(x, 0, i());
                }
            }
        }

        /**
         * @brief Triggers the reparameterization of the implicit domain
         * at the current dimension.
         */
        void reparam()
        {
            if (k == M)
            {
                this->reparamTensorProduct();
            }
            else if (0 <= k && k < M)
            {
                if constexpr (M == 1)
                {
                    const uvector<real,0> x;
                    const uvector<int,0> pt_id;
                    const uvector<int,0> elem_id;
                    this->computeReferenceIntervals(x, pt_id);
                    this->process(x, elem_id, pt_id);
                }
                else // if constexpr (1 < M)
                {
                    // Recursive call until reaching M == 1 or tensor product case.
                    this->base.reparam();
                }
            }

        }

        /**
         * @brief Given a list of polynomials, generates a class instance wrapped.
         * 
         * @param _polys Vector of polynomials to be evaluated.
         * @param _order Order of the reparameterization (number of points
         *        per direction in each reparameterization cell).
         * @return Class instance wrapped in a shared pointer.
         */
        static std::shared_ptr<ImplicitPolyReparam<N,N,S>>
        create(const std::vector<const xarray<real,N> *> &_polys, const int _order)
        {
            assert(!_polys.empty());

            const auto quad = std::make_shared<ImplicitPolyReparam<N,N,S>>();

            for(const auto *poly : _polys)
            {
                assert(poly != nullptr);
                const auto mask = algoim::detail::nonzeroMask(*poly, booluarray<N,ALGOIM_M>(true));
                if (algoim::detail::maskEmpty(mask))
                {
                    if (0.0 < bernstein::evalBernsteinPoly<N>(*poly, uvector<real,N>(0.5)))
                        return std::make_shared<ImplicitPolyReparam<N,N,S>>(); // Inactive domain.
                }
                else
                    quad->phi.push_back(*poly, mask);
            }
            quad->setOrder(_order);
            quad->build(true, false);

            return quad;
        }


        /**
         * @brief Reparameterizes the domain defined implicitly by a list of polynomials.
         * The interior of the domain is the subregion where all the polynomials are
         * negative at the same time.
         * 
         * @param _polys Vector of polynomials definining the domain.
         * @param _order Order of the reparameterization (number of points
         *        per direction in each reparameterization cell).
         * @return Vector of Lagrange elements that reparameterize
         *        either the surface or the volume.
         */
        static std::vector<std::shared_ptr<algoim::detail::ReparamElemType<S?N-1:N,N>>>
        reparameterize(const std::vector<const xarray<real,N> *> &_polys, const int _order)
        {
            const auto manager = create(_polys, _order); 
            manager->reparam();
            return manager->reparam_elems.reparam_elems;
        }

    };

    template<int N,bool S> struct ImplicitPolyReparam<0,N,S> {};

    /**
     * @brief Extracts the wirebasket of a Bezier domain reparameterization.
     * 
     * @tparam N Parametric dimension of the domain.
     * @tparam S True if the reparameterization corresponds only to the boundary.
     * @param _reparam Reparameterization whose wirebasket is extracted.
     * @param _polys Lisf of polynomials defining the domain.
     * @return Generated wirebasket reparameterization.
     */
    template<int N, bool S = false>
    std::vector<std::shared_ptr<algoim::lagrange::LagrangeTP<1,N>>>
    extractWirebasket(
        const std::vector<std::shared_ptr<algoim::detail::ReparamElemType<S?N-1:N,N>>> &_reparam,
        const std::vector<const xarray<real,N> *> &_polys)
    {
        static constexpr real tol = 1.0e4 * std::numeric_limits<real>::epsilon();

        assert(!_polys.empty());

        const auto check_face_pt = [&_polys](const uvector<real,N> &_pt) -> bool
        {
            for(const auto *poly : _polys)
            {
                if (std::abs(bernstein::evalBernsteinPoly(*poly, _pt)) < tol)
                    return true;
            }
            return false;
        };

        const auto check_internal_pt = [&_polys](const uvector<real,N> &_pt) -> bool
        {
            const int n_zeros = N == 2 ? 1 : 2;
            if (_polys.size() < n_zeros)
                return false;

            int counter = 0;
            for(const auto *poly : _polys)
            {
                if (std::abs(bernstein::evalBernsteinPoly(*poly, _pt)) < tol && ++counter == n_zeros)
                    return true;
            }
            return false;
        };

        const HyperRectangle<real, N> unit_domain(0.0, 1.0);
        return algoim::detail::extractWirebasket<S?N-1:N,N,S>(_reparam, unit_domain, check_face_pt, check_internal_pt);
    }

} // namespace detail

/**
 * @brief Reparameterizes a Bezier implicit function in the unit hypercube domain.
 * 
 * @tparam N Parametric dimension of the function.
 * @tparam S Flag indicating if the reparameterization must
 *         be performed only for the levelset surface (true), i.e.,
 *         the manifold where the Bezier function is equal to 0,
 *         or the volume (false), i.e., the subregion where
 *         the Bezier function is negative.
 * @param _bzr Bezier implicit function to reparameterize.
 * @param _order Order of the reparameterization (number of points
 *        per direction in each reparameterization cell).
 * @return Vector of Lagrange elements that reparameterize
 *        either the surface or the volume.
 */
template<int N, bool S = false>
std::vector<std::shared_ptr<algoim::detail::ReparamElemType<S?N-1:N,N>>>
reparam(const xarray<real, N> &_bzr, const int _order)
{
    std::vector<const xarray<real,N> *> polys;
    polys.push_back(&_bzr);

    return detail::ImplicitPolyReparam<N,N,S>::reparameterize(polys, _order);
}

/**
 * @brief Reparameterizes a domain defined by two Bezier implicit functions
 * in the unit hypercube domain.
 * 
 * @tparam N Parametric dimension of the function.
 * @tparam S Flag indicating if the reparameterization must
 *         be performed only for the levelset surface (true), i.e.,
 *         the manifold where the either of the two Bezier functions are
 *         equal to 0, or the subregion (false) between those surfaces,
 *         where both Bezier functions are negative.
 * @param _bzr0 First Bezier implicit function defining the domain.
 * @param _bzr1 Second Bspline implicit function defining the domain.
 * @param _order Order of the reparameterization (number of points
 *        per direction in each reparameterization cell).
 * @return Vector of Lagrange or Bezier elements that reparameterize
 *        either the surface or the volume.
 */
template<int N, bool S = false>
std::vector<std::shared_ptr<algoim::detail::ReparamElemType<S?N-1:N,N>>>
reparam(const xarray<real, N> &_bzr0, const xarray<real, N> &_bzr1, const int _order)
{
    std::vector<const xarray<real,N> *> polys;
    polys.push_back(&_bzr0);
    polys.push_back(&_bzr0);

    return detail::ImplicitPolyReparam<N,N,S>::reparameterize(polys, _order);
}

/**
 * @brief Reparameterizes a Bezier implicit function in the unit hypercube domain
 * creating only the edges wirebasket.
 * 
 * @tparam N Parametric dimension of the function.
 * @tparam S Flag indicating if the reparameterization must
 *         be performed only for the levelset surface (true), i.e.,
 *         the manifold where the Bezier function is equal to 0,
 *         or the volume (false), i.e., the subregion where
 *         the Bezier function is negative.
 * @param _bzr Bezier implicit function to reparameterize.
 * @param _order Order of the reparameterization (number of points
 *        per direction in each reparameterization cell).
 * @return Vector of Lagrange elements that reparameterize
 *        either the surface or the volume's wirebasket.
 */
template<int N, bool S = false>
std::vector<std::shared_ptr<algoim::lagrange::LagrangeTP<1,N>>>
reparamWirebasket(const xarray<real, N> &_bzr, const int _order)
{
    const auto rep = reparam<N,S>(_bzr, _order);
    std::vector<const xarray<real,N> *> polys;
    polys.push_back(&_bzr);
    return algoim::bezier::detail::extractWirebasket<N,S>(rep, polys);
}

/**
 * @brief Reparameterizes a domain defined by two Bezier implicit functions
 * in the unit hypercube domain creating only the edges wirebasket.
 * 
 * @tparam N Parametric dimension of the function.
 * @tparam S Flag indicating if the reparameterization must
 *         be performed only for the levelset surface (true), i.e.,
 *         the manifold where the either of the two Bezier functions are
 *         equal to 0, or the subregion (false) between those surfaces,
 *         where both Bezier functions are negative.
 * @param _bzr0 First Bezier implicit function defining the domain.
 * @param _bzr1 Second Bspline implicit function defining the domain.
 * @param _order Order of the reparameterization (number of points
 *        per direction in each reparameterization cell).
 * @return Vector of Lagrange or Bezier elements that reparameterize
 *        either the surface or the volume's wirebasket.
 */
template<int N, bool S = false>
std::vector<std::shared_ptr<algoim::lagrange::LagrangeTP<1,N>>>
reparamWirebasket(const xarray<real, N> &_bzr0, const xarray<real, N> &_bzr1, const int _order)
{
    const auto rep = reparam<N,S>(_bzr0, _bzr1, _order);

    std::vector<const xarray<real,N> *> polys;
    polys.push_back(&_bzr0);
    polys.push_back(&_bzr0);

    return algoim::bezier::detail::extractWirebasket<N,S>(rep, polys);
}

} // namespace algoim::bezier

#endif // ALGOIM_REPARAMETERIZATION_BEZIER_H
