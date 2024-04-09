#ifndef ALGOIM_REPARAMETERIZATION_GENERAL_H
#define ALGOIM_REPARAMETERIZATION_GENERAL_H

/* High-order accurate reparameterization algorithms for implicitly defined domains in hyperrectangles,
   based on the quadrature algorithms implemented in ImplicitIntegral and described in the paper:
    - R. I. Saye, High-Order Quadrature Methods for Implicitly Defined Surfaces and Volumes in Hyperrectangles,
      SIAM Journal on Scientific Computing, 37(2), A993-A1019 (2015),
      http://dx.doi.org/10.1137/140966290 */

#include "reparameterization_common.hpp"

#include "quadrature_general.hpp"

#include "bernstein.hpp"
#include "lagrange.hpp"
#include "bspline.hpp"
#include "uvector.hpp"

#include <memory>
#include <type_traits>

namespace algoim::general
{

namespace detail
{

    /**
     * @brief M-dimensional reparameterization of an N-dimensional function restricted to given
     * implicitly defined domains.
     * This is just a modification of the class ImplicitIntegral (quadrature_general.hpp)
     * for generating reparameterizations.
     * 
     * @tparam M Parametric dimension of the current sub-domain.
     * @tparam N Parametric dimension of the final domain.
     * @tparam Phi type of the implicit function to reparameterize.
     * @tparam F Type of auxiliary data structure for storing either the generated
     *         reparameterization (when T == true), or the higher-dimension
     *         class instance.
     * @tparam S Flag indicating if the reparameterization must
     *         be performed only for the levelset surface (true), i.e.,
     *         the manifold where the any of the polynomials is equal to 0,
     *         or the subregion volume (false) between those manifolds
     *         where all the polynomials are negative.
     * @tparam T Flag indicating if this one is the highest dimension last class instance,
     *         who is actually in charge of storing the generated reparameterization.
     */
    template<int M, int N, typename Phi, typename F, bool S, bool T=true>
    struct ImplicitReparam
    {
        static constexpr int n_max_subdivs = 16;

        const Phi& phi;
        F& f;
        const uvector<bool,N> free;
        std::array<PsiCode<N>,1 << (N - 1)> psi;
        int psiCount;
        const HyperRectangle<real,N> xrange;
        const int p;
        int e0;
        uvector<Interval<N>,N> xint;

        /// Reference domains intervals (defined as roots intervals)
        std::map<uvector<int,N>, algoim::detail::RootsIntervals<N>, util::UvectorCompare<int, N>> ref_intervals;
        /// Points at which the reference domains intervals were computed.
        std::map<uvector<int,N>, uvector<real,N>, util::UvectorCompare<int, N>> ref_intervals_points;

        // Prune the given set of functions by checking for the existence of the interface. If a function is
        // uniformly positive or negative and is consistent with specified sign, it can be removed. If a
        // function is uniformly positive or negative but inconsistent with specified sign, the domain of
        // integration is empty.
        bool prune()
        {
            for (int i = 0; i < psiCount; )
            {
                for (int dim = 0; dim < N; ++dim)
                    if (!free(dim))
                        xint(dim).alpha = xrange.side(psi[i].side(dim))(dim);
                Interval<N> res = phi(xint);
                if (res.uniformSign())
                {
                    if ((res.alpha >= 0.0 && psi[i].sign() >= 0) || (res.alpha <= 0.0 && psi[i].sign() <= 0))
                    {
                        --psiCount;
                        std::swap(psi[i], psi[psiCount]);
                    }
                    else
                        return false;
                }
                else
                    ++i;
            }
            return true;
        }

        /**
         * @brief Clears recursively the map of reparameterization elements.
         * It clears the map of reparameterization elements of the highest dimension.
         */
        void clearReparamElemsMap()
        {
            f.clearReparamElemsMap();
        }

        /**
         * @brief Clears recursively the reference intervals.
         * It clears the intervals for the current dimension and higher ones.
         */
        void clearReferenceIntervals()
        {
            this->ref_intervals.clear();
            this->ref_intervals_points.clear();
            this->f.clearReferenceIntervals();
        }

        /**
         * @brief Computes the tolerance to be used in calculations.
         * 
         * @return Computed tolerance.
         */
        real computeTolerance() const
        {
            constexpr real ref_tol = 10.0 * std::numeric_limits<real>::epsilon();
            const real tol = std::max(ref_tol, ref_tol * xrange.extent(e0));
            return tol;
        }

        /**
         * @brief Restricts the coordinates of the point @p _x according
         * to the @p _psi_id-th face restriction.
         * 
         * @param _psi_id Id of the restriction.
         * @param _x Point to constrain.
         */
        void setPsiBounds(const int _psi_id, uvector<real, N> &_x) const
        {
            for (int dim = 0; dim < N; ++dim)
                if (!free(dim))
                    _x(dim) = xrange.side(psi[_psi_id].side(dim))(dim);
        }

        /**
         * @brief Computes the roots of the function phi at point @p _x, 
         * along the direction e0, and under the constrains of the
         * @p _psi_id-th face restriction.
         * 
         * @param _x Point at which the roots are computed.
         * @param _psi_id Id of the restriction.
         * @return Computed roots.
         */
        std::vector<real> computeRoots(uvector<real,N> _x, const int _psi_id) const
        {
            this->setPsiBounds(_psi_id, _x);

            // x is now valid in all variables except e0
            std::vector<real> roots;
            algoim::detail::rootFind<Phi,N>(phi, _x, e0, xrange.min(e0), xrange.max(e0), roots, M > 1);

            return roots;
        }

        /**
         * @brief Checks if the interval defined by the coordinates @p _x0 and @p _x1
         * along e0 is active or not.
         * To be active, the interval must not have zero length and the function phi
         * evaluated inside the interval must be negative.
         * 
         * 
         * @param _x Partially restricted point at which the function phi is evaluated.
         * @param _x0 First end of the interval along e0.
         * @param _x1 Second end of the interval along e0.
         * @return True if the interval is active, false otherwise.
         */
        bool checkActiveInterval(uvector<real,N> &_x, const real _x0, const real _x1) const
        {
            const real tol = this->computeTolerance();

            if ((_x1 - _x0) < tol)
                return false;

            bool okay = true;
            _x(e0) = 0.5 * (_x0 + _x1);
            for (int j = 0; j < psiCount && okay; ++j)
            {
                this->setPsiBounds(j, _x);
                okay &= phi(_x) > 0.0 ? (psi[j].sign() >= 0) : (psi[j].sign() <= 0);
            }
            return okay;
        }

        /**
         * @brief Computes all the intervals (between consecutive roots)
         * at point @p _x along direction e0.
         * 
         * @param _x Point at which intervals are computed.
         * @return Computed intervals.
         */
        algoim::detail::RootsIntervals<N> computeAllIntervals(uvector<real,N> _x) const
        {
            algoim::detail::RootsIntervals<N> intervals;
            intervals.point = _x;
            auto &roots = intervals.roots;

            const real x0 = xrange.min(e0);
            const real x1 = xrange.max(e0);

            intervals.addRoot(x0, -1);
            intervals.addRoot(x1, -1);

            for (int i = 0; i < psiCount; ++i)
            {
                for(const auto r : this->computeRoots(_x, i))
                    intervals.addRoot(r, i);
            }

            // In rare cases, degenerate segments can be found, filter out with a tolerance
            const real tol = this->computeTolerance();
            intervals.adjustRoots(tol, x0, x1);

            if (!S)
            {
                const auto n_int = intervals.getNumRoots() - 1;
                assert(0 < n_int);

                intervals.active_intervals.resize(n_int);

                for(int i = 0; i < n_int; ++i)
                {
                    const real x0 = intervals.roots[i];
                    const real x1 = intervals.roots[i+1];
                    intervals.active_intervals[i] = this->checkActiveInterval(_x, x0, x1);
                }
            }

            return intervals;
        }

        /**
         * @brief Computes recursively the reference intervals for the current
         * dimension and the ones above.
         * 
         * @param _x Point at which the intervals are computed.
         * @param _elem_tid Element tensor index.
         */
        void computeReferenceIntervals(uvector<real,N> _x, uvector<int,N> _elem_tid)
        {
            if constexpr (M == 1)
            {
                f.clearReferenceIntervals();
            }

            const auto intervals = this->computeAllIntervals(_x);
            this->ref_intervals.emplace(_elem_tid, intervals);
            this->ref_intervals_points.emplace(_elem_tid, _x);


            // Loop over segments of divided interval
            const auto n_int = static_cast<int>(intervals.active_intervals.size());
            for(int i = 0; i < n_int; ++i)
            {
                if (!intervals.active_intervals[i])
                    continue;

                const real x0 = intervals.roots[i];
                const real x1 = intervals.roots[i+1];
                    
                _elem_tid(e0) = i;
                _x(e0) = 0.5 * (x0 + x1);
                f.computeReferenceIntervals(_x, _elem_tid);
            } // i
        }

        /**
         * @brief Reparameterizes the current domain to be the entire M-dimensional cube.
         */
        void tensorProductReparam()
        {
            if constexpr (!S)
            {
                if constexpr (T)
                {
                    using E = algoim::detail::ReparamElemType<M,N>;
                    if constexpr (M == N)
                    {
                        const auto cell = std::make_shared<E>(this->xrange, this->p);
                        f.reparam_elems.push_back(cell);
                    }
                    else
                    {
                        static_assert((M + 1) == N, "Invalid dimension");

                        int fix_dir{-1};
                        int side{-1};
                        for (int dim = 0; dim < N; ++dim)
                        {
                            if (!free(dim))
                            {
                                fix_dir = dim;
                                side = this->psi[0].side(fix_dir);
                                break;
                            }
                        }

                        const auto cell = std::make_shared<E>(this->xrange, this->p, fix_dir, side);
                        f.reparam_elems.push_back(cell);
                    }
                }
                else // if (!T)
                {
                    uvector<real,N> x;
                    for (int dim = 0; dim < N; ++dim)
                    {
                        if (free(dim))
                            x(dim) = xrange.midpoint(dim);
                    }
                    f.clearReferenceIntervals();
                    f.computeReferenceIntervals(x, 0);

                    for (MultiLoop<M> i(0,this->p); ~i; ++i)
                    {
                        uvector<real,N> x;
                        uvector<int,N> pt_tid;
                        for (int dim = 0, k = 0; dim < N; ++dim)
                        {
                            if (free(dim))
                            {
                                x(dim) = xrange.min(dim) + xrange.extent(dim) * bernstein::modifiedChebyshevNode(i(k), this->p);
                                pt_tid(dim) = i(k);
                                ++k;
                            }
                        }
                        f.process(x, 0, pt_tid);
                    }
                }
            }
        }

        /**
         * @brief Computes all the intervals (between consecutive roots)
         * at point @p _x along direction e0, taking as reference the provided
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
        algoim::detail::RootsIntervals<N>
        computeSimilarIntervals(const algoim::detail::RootsIntervals<N> &_ref_intervals,
                                  uvector<real,N> _x) const
        {
            algoim::detail::RootsIntervals<N> intervals;
            intervals.point = _x;

            const auto n_roots = _ref_intervals.getNumRoots();
            if (n_roots == 2)
            {
                if constexpr (S)
                    return intervals;
                else
                    return _ref_intervals;
            }

            const auto tol = this->computeTolerance();

            const auto x0 = xrange.min(e0);
            const auto x1 = xrange.max(e0);

            for (int i = 0; i < psiCount; ++i)
            {
                std::vector<real> roots_i;
                for (int j = 0; j < n_roots; ++j)
                {
                    const auto psi_id = _ref_intervals.func_ids[j];
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
                [tol,x0,x1](const auto &_r)
                {
                    return std::abs(_r - x0) < tol || std::fabs(_r - x1) < tol;
                });
                new_roots_i.erase(it, new_roots_i.end());
                const auto n_i_new = new_roots_i.size();

                if (n_i_new < n_i)
                {
                    this->setPsiBounds(i, _x);
                    _x(e0) = x0;
                    const auto root_0 = std::abs(this->phi(_x)) < tol;

                    _x(e0) = x1;
                    const auto root_1 = std::abs(this->phi(_x)) < tol;

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
                            auto xx = _ref_intervals.point;
                            this->setPsiBounds(i, xx);
                            xx(e0) = 0.5 * (x0 + roots_i.front());
                            const auto sign = this->phi(xx) > 0;

                            std::sort(new_roots_i.begin(), new_roots_i.end());
                            const real xmid = new_roots_i.empty() ? x1 : new_roots_i.front();
                            _x(e0) = 0.5 * (x0 + xmid);
                            const auto new_sign = this->phi(_x) > 0;

                            new_roots_i.push_back(sign == new_sign ? x1 : x0);
                        }
                    }
                }

                if (new_roots_i.size() != n_i) // Backup strategy.
                    new_roots_i = roots_i;

                for(const auto r : new_roots_i)
                    intervals.addRoot(r, i);
            }

            if constexpr (!S)
            {
                intervals.addRoot(x0, -1);
                intervals.addRoot(x1, -1);
                intervals.active_intervals = _ref_intervals.active_intervals;
                assert(intervals.getNumRoots() == _ref_intervals.getNumRoots());
            }

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
        algoim::detail::RootsIntervals<N>
        computeIntervals(uvector<real,N> _x, uvector<int,N> _elem_tid) const
        {
            const auto &ref_int = this->ref_intervals.at(_elem_tid);
            if constexpr (M == 1)
                return ref_int;
            else // if (1 < M)
                return computeSimilarIntervals(ref_int, _x);
        }

        /**
         * @brief This method performs the reparameterization in a recursive way by
         * computing the points coordinates along the different dimensions.
         * 
         * @param _x Partially computed point to be used in the reparameterization.
         * @param _elem_tid (Partial) tensor index of the element being reparameterized.
         * @param _pt_tid (Partial) tensor index of the reparameterization point.
         */
        void process(uvector<real,N> _x, uvector<int,N> _elem_tid, uvector<int,N> _pt_tid)
        {
            if constexpr (M == 1)
            {
                this->computeReferenceIntervals(_x, _elem_tid);
            }

            const auto intervals = this->computeIntervals(_x, _elem_tid);

            if constexpr (S)
            {
                static_assert(T, "Invalid dimension.");

                for(const auto root : intervals.roots)
                {
                    _x(e0) = root;
                    _elem_tid(e0) = 0;
                    const auto srf_pt_tid = remove_component(_pt_tid, e0);
                    f.process(_x, _elem_tid, srf_pt_tid);
                }
            }
            else
            {
                // Loop over segments of divided interval
                const auto n_int = static_cast<int>(intervals.active_intervals.size());
                for(int i = 0; i < n_int; ++i)
                {
                    if (!intervals.active_intervals[i])
                        continue;

                    const real x0 = intervals.roots[i];
                    const real x1 = intervals.roots[i+1];
                    
                    _elem_tid(e0) = i;

                    for (int j = 0; j < this->p; ++j)
                    {
                        _x(e0) = x0 + (x1 - x0) * bernstein::modifiedChebyshevNode(j, this->p);
                        _pt_tid(e0) = j;
                        if constexpr (T && M < N) // Face
                        {
                            int fix_dir{-1};
                            for (int dim = 0; dim < N; ++dim)
                            {
                                if (!free(dim))
                                    fix_dir = dim;
                            }
                            this->setPsiBounds(0, _x);


                            const auto face_pt_tid = remove_component(_pt_tid, fix_dir);
                            f.process(_x, _elem_tid, face_pt_tid);
                        }
                        else
                        {
                            f.process(_x, _elem_tid, _pt_tid);
                        }
                    }
                } // i
            }
        }

        // Main calling engine; parameters with underscores are copied upon entry but modified internally in the ctor
        ImplicitReparam(const Phi& phi, F& f, const uvector<bool,N>& free, const std::array<PsiCode<N>,1 << (N-1)>& psi_, int psiCount_, const HyperRectangle<real,N>& xrange, int p, int level = 0)
            : phi(phi), f(f), free(free), psi(psi_), psiCount(psiCount_), xrange(xrange), p(p)
        {
            // For the one-dimensional base case, evaluate the bottom-level integral.
            if constexpr (M == 1)
            {
                for (int dim = 0; dim < N; ++dim)
                    if (free(dim))
                        e0 = dim;
                process(real(0.0), 0, 0);
                return;
            }

            // Establish interval bounds for prune() and remaining part of ctor.
            for (int dim = 0; dim < N; ++dim)
            {
                if (free(dim))
                {
                    xint(dim) = Interval<N>(xrange.midpoint(dim), set_component<real,N>(0.0, dim, 1.0));
                    Interval<N>::delta(dim) = xrange.extent(dim)*0.5;
                }
                else
                {
                    xint(dim) = Interval<N>(real(0.0)); // xint(dim).delta will be set per psi function
                    Interval<N>::delta(dim) = real(0.0);
                }
            }

            // Prune list of psi functions: if prune procedure returns false, then the domain of integration is empty.
            if (!prune())
                return;

            // If all psi functions were pruned, then the volumetric integral domain is the entire hyperrectangle.
            if (psiCount == 0)
            {
                if (!S)
                    tensorProductReparam();
                return;
            }

            // Among all monotone height function directions, choose the one that makes the associated height function look as flat as possible.
            // This is a modification to the criterion presented in [R. Saye, High-Order Quadrature Methods for Implicitly Defined Surfaces and
            // Volumes in Hyperrectangles, SIAM J. Sci. Comput., Vol. 37, No. 2, pp. A993-A1019, http://dx.doi.org/10.1137/140966290].
            e0 = -1;
            real max_quan = 0.0;
            for (int dim = 0; dim < N; ++dim)
                if (!free(dim))
                    xint(dim).alpha = xrange.side(psi[0].side(dim))(dim);
            uvector<Interval<N>,N> g = phi.grad(xint);
            for (int dim = 0; dim < N; ++dim)
            {
                if (free(dim) && std::abs(g(dim).alpha) > 1.001*g(dim).maxDeviation())
                {
                    real quan = std::abs(g(dim).alpha) * xrange.extent(dim);
                    if (quan > max_quan)
                    {
                        max_quan = quan;
                        e0 = dim;
                    }
                }
            }

            // Check compatibility with all implicit functions whilst simultaneously constructing new implicit functions.
            std::array<PsiCode<N>,1 << (N-1)> newPsi;
            int newPsiCount = 0;
            for (int i = 0; i < psiCount; ++i)
            {
                // Evaluate gradient in an interval
                for (int dim = 0; dim < N; ++dim)
                    if (!free(dim))
                        xint(dim).alpha = xrange.side(psi[i].side(dim))(dim);
                uvector<Interval<N>,N> g = phi.grad(xint);

                // Determine if derivative in e0 direction is bounded away from zero.
                bool directionOkay = e0 != -1 && g(e0).uniformSign();
                if (!directionOkay)
                {
                    if (level < n_max_subdivs)
                    {
                        // Direction is not a good one, divide the domain into two along the biggest free extent
                        real maxext = 0.0;
                        int ind = -1;
                        for (int dim = 0; dim < N; ++dim)
                        {
                            if (free(dim))
                            {
                                real ext = xrange.extent(dim);
                                if (ext > maxext)
                                {
                                    maxext = ext;
                                    ind = dim;
                                }
                            }
                        }
                        assert(ind >= 0);
                        real xmid = xrange.midpoint(ind);
                        ImplicitReparam<M,N,Phi,F,S,T>(phi, f, free, psi, psiCount, HyperRectangle<real,N>(xrange.min(), set_component(xrange.max(), ind, xmid)), p, level + 1);
                        this->clearReparamElemsMap();

                        ImplicitReparam<M,N,Phi,F,S,T>(phi, f, free, psi, psiCount, HyperRectangle<real,N>(set_component(xrange.min(), ind, xmid), xrange.max()), p, level + 1);
                        return;
                    }
                    else
                    {
                        // Halt subdivision because we have recursively subdivided too deep; evaluate level set functions at
                        // the centre of box and check compatibility with signs.
                        uvector<real,N> xpoint = xrange.midpoint();
                        bool okay = true;
                        for (int j = 0; j < static_cast<int>(psi.size()) && okay; ++j)
                        {
                            for (int dim = 0; dim < N; ++dim)
                                if (!free(dim))
                                    xpoint(dim) = xrange.side(psi[j].side(dim))(dim);
                            okay &= phi(xpoint) >= 0.0 ? (psi[j].sign() >= 0) : (psi[j].sign() <= 0);
                        }
                        if (okay)
                        {
                            if (S)
                                assert(M == N);
                            else
                            {
                                this->tensorProductReparam();
                            }
                        }
                        return;
                    }
                }
                
                // Direction is okay - build restricted level set functions and determine the appropriate signs
                int bottomSign, topSign;
                algoim::detail::determineSigns<S>(g(e0).alpha > 0.0, psi[i].sign(), bottomSign, topSign);
                newPsi[newPsiCount++] = PsiCode<N>(psi[i], e0, 0, bottomSign);
                newPsi[newPsiCount++] = PsiCode<N>(psi[i], e0, 1, topSign);
                assert(newPsiCount <= 1 << (N - 1));
            }

            // Dimension reduction call
            assert(e0 != -1);
            ImplicitReparam<M-1,N,Phi,ImplicitReparam<M,N,Phi,F,S,T>,false,false>(phi, *this, set_component(free, e0, false), newPsi, newPsiCount, xrange, p);
        }
    };


    // Partial specialization on M=0 as a dummy base case for the compiler
    template<int N, typename Phi, typename F, bool S, bool T>
    struct ImplicitReparam<0,N,Phi,F,S, T>
    {
        ImplicitReparam(const Phi&, F&, const uvector<bool,N>&, const std::array<PsiCode<N>,1 << (N-1)>&, int, const HyperRectangle<real,N>&, int) {}
    };

    /**
     * @brief Extracts the wirebasket of a Bezier domain reparameterization.
     * 
     * @tparam F Type of the function to reparameterize.
     * @tparam M Parametric dimension of the reparameterization whose wirebasket is extracted.
     * @tparam N Physical dimension of the reparameterization whose wirebasket is extracted.
     * @tparam S True if the reparameterization corresponds only to the boundary.
     * @param _reparam Reparameterization whose wirebasket is extracted.
     * @param _phi Function defining the domain.
     * @param _domain Hypercube defining the limits of the function.
     * @return 
     */
    template<int M, int N, typename F, bool S = false>
    std::vector<std::shared_ptr<algoim::detail::ReparamElemType<1,N>>>
    extractWirebasket(
        const std::vector<std::shared_ptr<algoim::detail::ReparamElemType<M,N>>> &_reparam,
        const F &_phi,
        const HyperRectangle<real,N>& _domain)
    {
        static constexpr real tol = 1.0e4 * std::numeric_limits<real>::epsilon();

        const auto check_face_pt = [&_phi](const uvector<real,N> &_pt) -> bool
        {
            return std::abs(_phi(_pt)) < tol;
        };

        const auto check_internal_pt = [](const uvector<real,N> &_pt) -> bool
        {
            return false;
        };

        return algoim::detail::extractWirebasket<M,N,S>(_reparam, _domain, check_face_pt, check_internal_pt);
    }

} // detail


/**
 * @brief Reparameterizes the domain defined by an implicit function.
 * 
 * @tparam N Parametric dimension of the function.
 * @tparam F Function type.
 * 
 * @param _phi Function to reparameterize.
 * @param _domain Domain to reparameterize.
 * @param _order Order of the reparameterization (number of points
 *        per direction in each reparameterization cell).
 * @return Vector of Lagrange elements that reparameterize the
 *         domain (i.e., where @p _phi is negative).
 */
template<int N, typename F>
std::vector<std::shared_ptr<algoim::detail::ReparamElemType<N,N>>>
reparam(const F &_phi, const HyperRectangle<real,N>& _domain, const int _order)
{
    std::array<PsiCode<N>,1 << (N - 1)> psi;
    psi[0] = PsiCode<N>(0, -1);
    uvector<bool,N> free = true;

    using R = algoim::detail::ReparamElems<N,N>;
    R elems(_order);

    const detail::ImplicitReparam<N,N,F,R,false> impl(_phi, elems, free, psi, 1, _domain, _order);
    return elems.reparam_elems;
}

/**
 * @brief Reparameterizes the zero levelset surface of an implicit function.
 * 
 * @tparam N Parametric dimension of the function.
 * @tparam F Function type.
 * 
 * @param _phi Function to reparameterize.
 * @param _domain Domain to reparameterize.
 * @param _order Order of the reparameterization (number of points
 *        per direction in each reparameterization cell).
 * @return Vector of Lagrange elements that reparameterize the zero
 *         levelset (i.e., where @p _phi equals 0).
 */
template<int N, typename F>
std::vector<std::shared_ptr<algoim::detail::ReparamElemType<N-1,N>>>
reparamLevelset(const F &_phi, const HyperRectangle<real,N>& _domain, const int _order)
{
    std::array<PsiCode<N>,1 << (N - 1)> psi;
    psi[0] = PsiCode<N>(0, -1);
    uvector<bool,N> free = true;

    using R = algoim::detail::ReparamElems<N-1,N>;
    R elems(_order);

    const detail::ImplicitReparam<N,N,F,R,true> impl(_phi, elems, free, psi, 1, _domain, _order);
    return elems.reparam_elems;
}

/**
 * @brief Reparameterizes a face of domain defined by an implicit function.
 * It reparameterizes one of the 2^dim faces of the domain.
 * The faces are defined as _domain(_side)(_dir).
 * 
 * @tparam N Parametric dimension of the function.
 * @tparam F Function type.
 * 
 * @param _phi Function to reparameterize.
 * @param _domain Domain to reparameterize.
 * @param _order Order of the reparameterization (number of points
 *        per direction in each reparameterization cell).
 * @param _dir Constant direction of the face to reparameterize.
 * @param _side Side of the face to reparameterize.
 * @return Vector of Lagrange elements that reparameterize the
 *         domain (i.e., where @p _phi is negative).
 */
template<int N, typename F>
std::vector<std::shared_ptr<algoim::detail::ReparamElemType<N-1,N>>>
reparamFace(const F &_phi, const HyperRectangle<real,N>& _domain, const int _order,
             const int _dir, const int _side)
{
    assert(0 <= _dir && _dir < N);
    assert(_side == 0 || _side == 1);

    std::array<PsiCode<N>,1 << (N - 1)> psi;
    psi[0] = PsiCode<N>(0, -1);
    uvector<bool,N> free = true;

    psi[0] = PsiCode<N>(set_component<int,N>(0, _dir, _side), -1);
    free(_dir) = false;

    using R = algoim::detail::ReparamElems<N-1,N>;
    R elems(_order);

    const detail::ImplicitReparam<N-1,N,F,R,false> impl(_phi, elems, free, psi, 1, _domain, _order);

    return elems.reparam_elems;
}

/**
 * @brief Reparameterizes the domain defined by an implicit function
 * creating only the edges wirebasket.
 * 
 * @tparam N Parametric dimension of the function.
 * @tparam F Function type.
 * 
 * @param _phi Function to reparameterize.
 * @param _domain Domain to reparameterize.
 * @param _order Order of the reparameterization (number of points
 *        per direction in each reparameterization cell).
 * @return Vector of Lagrange elements that reparameterize the
 *         wirebasket of the domain (i.e., where @p _phi goes
 *         from positive to negative).
 */
template<int N, typename F>
std::vector<std::shared_ptr<algoim::detail::ReparamElemType<1,N>>>
reparamWirebasket(const F &_phi, const HyperRectangle<real,N>& _domain, const int _order)
{
    const auto rep = reparam<N,F>(_phi, _domain, _order);
    return detail::extractWirebasket<N,N,F,false>(rep, _phi, _domain);
}

/**
 * @brief Reparameterizes the zero levelset surface of an implicit function
 * creating only the edges wirebasket.
 * 
 * @tparam N Parametric dimension of the function.
 * @tparam F Function type.
 * 
 * @param _phi Function to reparameterize.
 * @param _domain Domain to reparameterize.
 * @param _order Order of the reparameterization (number of points
 *        per direction in each reparameterization cell).
 * @return Vector of Lagrange elements that reparameterize the zero
 *         wirebasket of the levelset (i.e., where @p _phi equals 0).
 */
template<int N, typename F>
std::vector<std::shared_ptr<algoim::detail::ReparamElemType<1,N>>>
reparamLevelsetWirebasket(const F &_phi, const HyperRectangle<real,N>& _domain, const int _order)
{
    const auto rep = reparamLevelset<N,F>(_phi, _domain, _order);
    return detail::extractWirebasket<N-1,N,F,true>(rep, _phi, _domain);
}

/**
 * @brief Reparameterizes a face of domain defined by an implicit function.
 * It reparameterizes one of the 2^dim faces of the domain creating only the edges wirebasket.
 * The faces are defined as _domain(_side)(_dir).
 * 
 * @tparam N Parametric dimension of the function.
 * @tparam F Function type.
 * 
 * @param _phi Function to reparameterize.
 * @param _domain Domain to reparameterize.
 * @param _order Order of the reparameterization (number of points
 *        per direction in each reparameterization cell).
 * @param _dir Constant direction of the face to reparameterize.
 * @param _side Side of the face to reparameterize.
 * @return Vector of Lagrange elements that reparameterize the
 *         wirebasket of the domain (i.e., the boundary where @p _phi
 *         goes from positive to negative).
 */
template<int N, typename F>
std::vector<std::shared_ptr<algoim::detail::ReparamElemType<1,N>>>
reparamFaceWirebasket(const F &_phi, const HyperRectangle<real,N>& _domain, const int _order,
             const int _dir, const int _side)
{
    const auto rep = reparamFace<N,F>(_phi, _domain, _order, _dir, _side);
    return detail::extractWirebasket<N-1,N,F,false>(rep, _phi, _domain);
}

} // namespace algoim::general

#endif // ALGOIM_REPARAMETERIZATION_GENERAL_H
