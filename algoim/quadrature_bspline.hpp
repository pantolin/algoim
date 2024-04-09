#ifndef ALGOIM_QUADRATURE_BSPLINE_H
#define ALGOIM_QUADRATURE_BSPLINE_H

/* Creates a quadrature for a tensor-product BSpline function. */

#include "real.hpp"
#include "uvector.hpp"
#include "hyperrectangle.hpp"
#include "bspline.hpp"
#include "bezier.hpp"
#include "quadrature_general_decomposition.hpp"
#include "quadrature_bezier.hpp"
#include "quadrature_element.hpp"

#include <algorithm>
#include <memory>
#include <map>

namespace algoim::bspline
{
    namespace detail
    {
        template<int N, int R>
        void getBezierIds(const BSplineTP<N, R> &_spline,
                            const BSplineTP<N, R> &_sub_spline,
                            std::vector<int> &_bezier_ids)
        {
            _bezier_ids.reserve(_bezier_ids.size() + prod(_sub_spline.getNumElemsDir()));

            const auto n_elems = prod(_sub_spline.getNumElemsDir());
            for(int bzr_id = 0; bzr_id < n_elems; ++bzr_id)
            {
                const auto bzr_tid = _sub_spline.getTensorIndex(bzr_id);
                const auto bzr_domain = _sub_spline.getElementDomain(bzr_tid);
                const auto parent_bzr_tid = _spline.findElement(bzr_domain.midpoint());
                const auto parent_bzr_id = _spline.getFlatIndex(parent_bzr_tid);
                _bezier_ids.push_back(parent_bzr_id);
            }
        }

        template<int N, int R>
        void getBezierIds(const BSplineTP<N, R> &_spline,
                            std::vector<std::shared_ptr<const BSplineTP<N, R>>> &_sub_splines,
                            std::vector<int> &_bezier_ids)
        {
            for(const auto &sub_spline : _sub_splines)
            {
                getBezierIds(_spline, *sub_spline, _bezier_ids);
            }
        }

    } // namespace detail

    template<int N>
    struct Quadrature
    {
        std::vector<int> int_bzrs;
        std::vector<int> ext_bzrs;
        std::map<int, std::shared_ptr<const quad::ElementQuadrature<N>>> cut_bzrs;
    };

    template<int N>
    std::shared_ptr<const Quadrature<N>>
    createQuadrature(const BSplineTP<N, 1> &_spline,
                      const int _qo,
                      const bool _iso_bounds)
    {
        const auto quad = std::make_shared<Quadrature<N>>();
        auto &int_bzrs = quad->int_bzrs;
        auto &ext_bzrs = quad->ext_bzrs;
        auto &cut_bzrs = quad->cut_bzrs;

        using SplinePtr = std::shared_ptr<const BSplineTP<N, 1>>;
        std::vector<SplinePtr> internals, externals, unknowns;
        decompose(_spline, internals, externals, unknowns);

        std::vector<int> bezier_ids;
        for(const auto &unknown : unknowns)
        {
            bezier_ids.clear();
            detail::getBezierIds(_spline, *unknown, bezier_ids);
            assert(bezier_ids.size() == 1);
            const auto bzr_id = bezier_ids.front();

            const auto bzr = unknown->transformToBezier();
            const auto elem_quad = bezier::createQuadrature(*bzr, _qo, _iso_bounds);

            if (elem_quad->cut)
            { 
                cut_bzrs.emplace(bzr_id, elem_quad);
            }
            else
            {
                if (elem_quad->interior)
                    int_bzrs.push_back(bzr_id);
                else
                    ext_bzrs.push_back(bzr_id);
            }
        }

        detail::getBezierIds(_spline, internals, int_bzrs);
        detail::getBezierIds(_spline, externals, ext_bzrs);

        std::sort(int_bzrs.begin(), int_bzrs.end());
        std::sort(ext_bzrs.begin(), ext_bzrs.end());

        return quad;
    }

    template<int N>
    std::shared_ptr<const Quadrature<N>>
    createQuadrature(const BSplineTP<N, 1> &_spline_0,
                      const BSplineTP<N, 1> &_spline_1,
                      const int _qo,
                      const bool _iso_bounds)
    {
        const auto quad = std::make_shared<Quadrature<N>>();
        auto &int_bzrs = quad->int_bzrs;
        auto &ext_bzrs = quad->ext_bzrs;
        auto &cut_bzrs = quad->cut_bzrs;

        using SplinePtr = std::shared_ptr<const BSplineTP<N, 1>>;
        std::vector<std::array<SplinePtr,2>> internals, externals, unknowns;
        decompose(_spline_0, _spline_1, internals, externals, unknowns);

        std::vector<int> bezier_ids;
        for(const auto &unknown : unknowns)
        {
            bezier_ids.clear();
            detail::getBezierIds(_spline_0, *unknown[0], bezier_ids);
            assert(bezier_ids.size() == 1);
            const auto bzr_id = bezier_ids.front();

            const auto bzr_0 = unknown[0]->transformToBezier();
            const auto bzr_1 = unknown[1]->transformToBezier();
            const auto elem_quad = bezier::createQuadrature(*bzr_0, *bzr_1, _qo, _iso_bounds);

            if (elem_quad->cut)
            {
                cut_bzrs.emplace(bzr_id, elem_quad);
            }
            else
            {
                if (elem_quad->interior)
                    int_bzrs.push_back(bzr_id);
                else
                    ext_bzrs.push_back(bzr_id);
            }
        }

        std::vector<SplinePtr> internals_0, externals_0;
        internals_0.reserve(internals.size());
        externals_0.reserve(externals.size());

        std::transform(internals.cbegin(), internals.cend(),
                       std::back_inserter(internals_0), 
                       [](const auto &_internal) { return _internal[0];});

        std::transform(externals.cbegin(), externals.cend(),
                       std::back_inserter(externals_0), 
                       [](const auto &_external) { return _external[0];});

        detail::getBezierIds(_spline_0, internals_0, int_bzrs);
        detail::getBezierIds(_spline_0, externals_0, ext_bzrs);

        std::sort(int_bzrs.begin(), int_bzrs.end());
        std::sort(ext_bzrs.begin(), ext_bzrs.end());

        return quad;
    }


} // namespace algoim::bspline

#endif // ALGOIM_QUADRATURE_BSPLINE_H
