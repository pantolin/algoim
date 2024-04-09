#ifndef ALGOIM_QUADRATURE_GENERAL_DECOMPOSITION_H
#define ALGOIM_QUADRATURE_GENERAL_DECOMPOSITION_H

#include "quadrature_general.hpp"
#include "hyperrectangle.hpp"
#include "real.hpp"
#include "uvector.hpp"
#include "bspline.hpp"
#include "cartesian_grid.hpp"
#include "quadrature_multipoly.hpp"

#include <memory>

namespace algoim::general
{

    /**
     * @brief Studies the sign of the function @p _phi in the given @p _subgrid.
     * 
     * Checks if an element is internal (the sign of @p _phi is negative
     * everywhere in the full domain of the @p _subgrid ),
     * external (the sign is positive), or it is unknown.
     * 
     * @tparam F Type of implicit function.
     * @tparam N Parametric dimension.
     * @param _phi Implicit function to study.
     * @param _subgrid Subgrid to query.
     * @param _internal_elements List of elements fully internal.
     * @param _external_elements List of elements fully external.
     * @param _unknown_elements List of elements that can be either internal or external.
     */
    template<int N, typename F>
    void
    decompose(const F &_phi,
              const SubGrid<N> &_subgrid,
              std::vector<int> &_internal_elements,
              std::vector<int> &_external_elements,
              std::vector<int> &_unknown_elements)
    {
        const auto domain = _subgrid.getDomain();

        uvector<Interval<N>,N> xint;
        for (int dir = 0; dir < N; ++dir)
        {
            const auto beta = set_component<real,N>(0.0, dir, 1.0);
            xint(dir) = Interval<N>(domain.midpoint(dir), beta);
            Interval<N>::delta(dir) = 0.5 * domain.extent(dir);
        }

        Interval<N> res = _phi(xint);
        if (res.uniformSign())
        {
            const auto elem_indices = _subgrid.getElementIndices();
            if (res.alpha < 0.0)
            {
                _internal_elements.insert(_internal_elements.end(), elem_indices.cbegin(), elem_indices.cend());
            }
            else // if (res.alpha >= 0.0)
            {
                _external_elements.insert(_external_elements.end(), elem_indices.cbegin(), elem_indices.cend());
            }
        }
        else
        {
            if (_subgrid.uniqueElement())
            {
                _unknown_elements.push_back(_subgrid.getElementIndices().front());
            }
            else
            {
                for(const auto &sub_grid : _subgrid.split())
                    decompose(_phi, *sub_grid, _internal_elements, _external_elements, _unknown_elements);
            }
        }
    }

} // namespace algoim::general

namespace algoim::bspline
{
    namespace detail
    {
    template<int N>
    class BSplineDomainDecomposition
    {
        using Spline = BSplineTP<N, 1>;
        using SplinePtr = std::shared_ptr<const Spline>;

    public:
        BSplineDomainDecomposition(const Spline &_phi)
        : BSplineDomainDecomposition(_phi.clone())
        {}

        BSplineDomainDecomposition(const SplinePtr _phi)
        :
        exterior(),
        interior(),
        unknown()
        {
            assert(_phi != nullptr);

            const auto n_elems = prod(_phi->getNumElemsDir());
            exterior.reserve(n_elems);
            interior.reserve(n_elems);
            unknown.reserve(n_elems);

            this->perform(_phi);
        }

        std::vector<SplinePtr> exterior;
        std::vector<SplinePtr> interior;
        std::vector<SplinePtr> unknown;

    private:

        void merge(const BSplineDomainDecomposition<N> &_dd)
        {
            exterior.insert(exterior.end(), _dd.exterior.cbegin(), _dd.exterior.end());
            interior.insert(interior.end(), _dd.interior.cbegin(), _dd.interior.end());
            unknown.insert(unknown.end(), _dd.unknown.cbegin(), _dd.unknown.end());
        }

        void perform(const SplinePtr _phi)
        {
            const auto change_sign = _phi->studyChangeOfSign();

            if (change_sign == -1)
            {
                interior.push_back(_phi);
            }
            else if (change_sign == 1)
            {
                exterior.push_back(_phi);
            }
            else if (_phi->isBezierLike())
            {
                const xarray<real, N> &coefs = _phi->getXarray();
                const auto mask = algoim::detail::nonzeroMask(coefs, booluarray<N,ALGOIM_M>(true));
                if (algoim::detail::maskEmpty(mask))
                {
                    if (0.0 < bernstein::evalBernsteinPoly<N>(coefs, uvector<real,N>(0.5)))
                    {
                        exterior.push_back(_phi);
                        return;
                    }
                }

                unknown.push_back(_phi);
            }
            else
            {
                const auto new_splines = _phi->split();

                for(int i = 0; i < 2; ++i)
                {
                    const BSplineDomainDecomposition<N> dd(new_splines[i]);
                    this->merge(dd);
                }
            }
        }
    };

    template<int N>
    class BSpline2DomainDecomposition
    {
        using Spline = BSplineTP<N, 1>;
        using SplinePtr = std::shared_ptr<const Spline>;

    public:

        BSpline2DomainDecomposition(const Spline &_phi0,
                                    const Spline &_phi1)
        : BSpline2DomainDecomposition(_phi0.clone(), _phi1.clone())
        {}

        BSpline2DomainDecomposition(const SplinePtr &_phi0,
                                    const SplinePtr &_phi1)
        :
        exterior(),
        interior(),
        unknown()
        {
            assert(_phi0 != nullptr);
            assert(_phi1 != nullptr);

            const auto n_elems = prod(_phi0->getNumElemsDir());
            for(int dir = 0; dir < N; ++dir)
                assert(_phi0->getNumElemsDir()(dir) == _phi1->getNumElemsDir()(dir));
            // TODO: to check that both splines have the same knot vectors.

            exterior.reserve(n_elems);
            interior.reserve(n_elems);
            unknown.reserve(n_elems);

            this->perform(_phi0, _phi1);
        }

        std::vector<std::array<SplinePtr, 2>> exterior;
        std::vector<std::array<SplinePtr, 2>> interior;
        std::vector<std::array<SplinePtr, 2>> unknown;

    private:

        void merge(const BSpline2DomainDecomposition<N> &_dd)
        {
            exterior.insert(exterior.end(), _dd.exterior.cbegin(), _dd.exterior.end());
            interior.insert(interior.end(), _dd.interior.cbegin(), _dd.interior.end());
            unknown.insert(unknown.end(), _dd.unknown.cbegin(), _dd.unknown.end());
        }

        void perform(const SplinePtr _phi0, const SplinePtr _phi1)
        {
            const auto change_sign0 = _phi0->studyChangeOfSign();
            const auto change_sign1 = _phi1->studyChangeOfSign();

            if (change_sign0 == -1 && change_sign1 == -1)
            {
                interior.push_back({_phi0, _phi1});
            }
            else if (change_sign0 == 1 && change_sign1 == 1)
            {
                exterior.push_back({_phi0, _phi1});
            }
            else if (_phi0->isBezierLike())
            {
                assert(_phi1->isBezierLike());

                for(const auto phi : {_phi0, _phi1})
                {
                    const xarray<real, N> &coefs = phi->getXarray();
                    const auto mask = algoim::detail::nonzeroMask(coefs, booluarray<N,ALGOIM_M>(true));
                    if (algoim::detail::maskEmpty(mask))
                    {
                        if (0.0 < bernstein::evalBernsteinPoly<N>(coefs, uvector<real,N>(0.5)))
                        {
                            exterior.push_back({_phi0, _phi1});
                            return;
                        }
                    }
                }

                unknown.push_back({_phi0, _phi1});
            }
            else
            {
                const auto new_splines0 = _phi0->split();
                const auto new_splines1 = _phi1->split();

                for(int i = 0; i < 2; ++i)
                {
                    const BSpline2DomainDecomposition<N> dd(new_splines0[i], new_splines1[i]);
                    this->merge(dd);
                }
            }
        }
    };

    } // namespace detail

    /**
     * @brief Splits the given @p _spline according to the sign of its image.
     * 
     * Checks if the image of @p _spline is fully internal (the sign is negative
     * everywhere in its full domain), external (the sign is positive), or it is unknown.
     * If it is internal or external, the spline is returned, otherwise (if unknown),
     * it is split according to the knot lines and the algorithm is applied recursively.
     * The algorithm stops at the Bezier level.
     * 
     * @tparam N Parametric dimension.
     * @param _spline Spline function to study.
     * @param _internal List of spline pieces that are fully internal.
     * @param _external List of spline pieces that are fully external.
     * @param _unknown List of spline elements (Beziers) that whose status is unknown,
     * i.e., they can be internal, external, or cut.
     */
    template<int N>
    void
    decompose(const BSplineTP<N, 1> &_spline,
              std::vector<std::shared_ptr<const BSplineTP<N, 1>>> &_internal,
              std::vector<std::shared_ptr<const BSplineTP<N, 1>>> &_external,
              std::vector<std::shared_ptr<const BSplineTP<N, 1>>> &_unknown)
    {
        const detail::BSplineDomainDecomposition<N> decomposer(_spline);
        _internal = decomposer.interior;
        _external = decomposer.exterior;
        _unknown = decomposer.unknown;
    }

    /**
     * @brief Splits the given @p _spline_0 and @p _spline_1 according to the sign of their images.
     * 
     * Checks if the image of both @p _spline_0 and @p _spline_1 is fully internal
     * (the signs are negative everywhere in their full domain), external (the signs
     * are positive), or it is unknown.
     * If they are internal or external, the splines are returned, otherwise (if unknown),
     * they are split according to the knot lines and the algorithm is applied recursively.
     * The algorithm stops at the Bezier level.
     * 
     * @tparam N Parametric dimension.
     * @param _spline_0 First spline function to study.
     * @param _spline_1 Second spline function to study.
     * @param _internal List of spline pieces that are fully internal.
     * @param _external List of spline pieces that are fully external.
     * @param _unknown List of spline elements (Beziers) that whose status is unknown,
     * i.e., they can be internal, external, or cut.
     */
    template<int N>
    void
    decompose(const BSplineTP<N, 1> &_spline_0,
              const BSplineTP<N, 1> &_spline_1,
              std::vector<std::array<std::shared_ptr<const BSplineTP<N, 1>>,2>> &_internal,
              std::vector<std::array<std::shared_ptr<const BSplineTP<N, 1>>,2>> &_external,
              std::vector<std::array<std::shared_ptr<const BSplineTP<N, 1>>,2>> &_unknown)
    {
        const detail::BSpline2DomainDecomposition<N> decomposer(_spline_0, _spline_1);
        _internal = decomposer.interior;
        _external = decomposer.exterior;
        _unknown = decomposer.unknown;
    }

} // namespace algoim::bspline

#endif // ALGOIM_QUADRATURE_GENERAL_DECOMPOSITION_H
