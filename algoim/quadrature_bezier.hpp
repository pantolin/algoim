#ifndef ALGOIM_QUADRATURE_BEZIER_H
#define ALGOIM_QUADRATURE_BEZIER_H

/* Creates a quadrature for a tensor-product BSpline function. */

#include "real.hpp"
#include "uvector.hpp"
#include "hyperrectangle.hpp"
#include "bezier.hpp"
#include "quadrature_element.hpp"
#include "quadrature_multipoly.hpp"

#include <memory>
#include <numeric>

namespace algoim::bezier
{
    template<int N>
    void postprocessIsoBndryQuadrature(quad::QuadratureIsoBoundary<N> &_quad,
                                       const int _iso_bdry_id)
    {
        _quad.cut = !_quad.points.empty();
        _quad.interior = false;

        if (_quad.cut)
        {
            // Checking if the full face is active.
            const real exact_area{1.0};
            const real approx_area = std::accumulate(_quad.weights.cbegin(), _quad.weights.cend(), real{0.0});
            constexpr real tolerance = std::numeric_limits<real>::epsilon();
            _quad.cut = std::abs(approx_area - exact_area) > tolerance;

            _quad.interior = !_quad.cut;
        }

        if (_quad.cut)
        {
            const auto n_points = static_cast<int>(_quad.points.size());
            _quad.fillNormaliso(n_points, _iso_bdry_id);
        }
        else
        {
            _quad.points.clear();
            _quad.weights.clear();
            _quad.normals.clear();
        }
    }

    template<int N>
    void
    createQuadratureIsoBounds(const xarray<real, N> &_bezier,
                              const int _qo,
                              quad::ElementQuadrature<N> &_elem_quad)
    {
        for(int dir = 0, iso_bdry_id = 0; dir < N; ++dir)
        {
            xarray<real,N-1> bzr_face(nullptr, remove_component(_bezier.ext(), dir));
            algoim_spark_alloc(real, bzr_face);

            for(int side = 0; side < 2; ++side, ++iso_bdry_id)
            {
                algoim::detail::restrictToFace(_bezier, dir, side, bzr_face);
                ImplicitPolyQuadrature<N-1> ipquad(bzr_face);

                auto &quad = _elem_quad.iso_bdry_quad[iso_bdry_id];

                const real const_val = static_cast<real>(side);
                ipquad.integrate(AlwaysGL, _qo, [&quad,&bzr_face,const_val,dir]
                (const uvector<real, N-1> &_x, const real _w)
                {
                    if (bernstein::evalBernsteinPoly(bzr_face, _x) < 0)
                    {
                        quad.points.push_back(add_component(_x, dir, const_val));
                        quad.weights.push_back(_w);
                    }
                });

                postprocessIsoBndryQuadrature(quad, iso_bdry_id);
            } // side
        } // dir
    }

    template<int N>
    void
    createQuadratureIsoBounds(const xarray<real, N> &_bezier_0,
                              const xarray<real, N> &_bezier_1,
                              const int _qo,
                              quad::ElementQuadrature<N> &_elem_quad)
    {
        for(int dir = 0, iso_bdry_id = 0; dir < N; ++dir)
        {
            xarray<real,N-1> bzr_face_0(nullptr, remove_component(_bezier_0.ext(), dir));
            xarray<real,N-1> bzr_face_1(nullptr, remove_component(_bezier_1.ext(), dir));
            algoim_spark_alloc(real, bzr_face_0);
            algoim_spark_alloc(real, bzr_face_1);

            for(int side = 0; side < 2; ++side, ++iso_bdry_id)
            {
                algoim::detail::restrictToFace(_bezier_0, dir, side, bzr_face_0);
                algoim::detail::restrictToFace(_bezier_1, dir, side, bzr_face_1);

                ImplicitPolyQuadrature<N-1> ipquad(bzr_face_0, bzr_face_1);

                auto &quad = _elem_quad.iso_bdry_quad[iso_bdry_id];

                const real const_val = static_cast<real>(side);
                ipquad.integrate(AlwaysGL, _qo, [&quad,&bzr_face_0,&bzr_face_1,const_val,dir]
                (const uvector<real, N-1> &_x, const real _w)
                {
                    if (bernstein::evalBernsteinPoly(bzr_face_0, _x) < 0 &&
                        bernstein::evalBernsteinPoly(bzr_face_1, _x) < 0)
                    {
                        quad.points.push_back(add_component(_x, dir, const_val));
                        quad.weights.push_back(_w);
                    }
                });

                postprocessIsoBndryQuadrature(quad, iso_bdry_id);
            } // side
        } // dir
    }


    template<int N>
    std::shared_ptr<const quad::ElementQuadrature<N>>
    createQuadrature(const BezierTP<N, 1> &_bezier,
                     const int _qo,
                     const bool _iso_bounds)
    {
        const auto elem_quad = std::make_shared<quad::ElementQuadrature<N>>();

        auto &bnd_quad = elem_quad->bdry_quads["levelset"];
        auto &quad = elem_quad->quad;

        const auto &bzr = _bezier.getXarray();

        ImplicitPolyQuadrature<N> ipquad(bzr);

        ipquad.integrate_surf(AlwaysGL, _qo, [&bnd_quad,&bzr](const uvector<real,N> &_x, const real _w, const uvector<real,N>& _wn)
        {
            bnd_quad.points.push_back(_x);
            bnd_quad.weights.push_back(_w);
            bnd_quad.normals.push_back(_wn / _w);

            // bnd_quad.normals.push_back(_wn);

            // TODO: I have to study better which is the output _wn.
            // I don't understand the difference between the
            // single and aggregated cases and it's crucial for this quantity.

            uvector<real,N> n = bernstein::evalBernsteinPolyGradient(bzr, _x);

            if (norm(n) > 0)
                n *= real(1.0) / norm(n);
            bnd_quad.normals.push_back(n);
        });

        elem_quad->cut = !bnd_quad.points.empty();

        if (elem_quad->cut)
        {
            elem_quad->interior = false;
            ipquad.integrate(AlwaysGL, _qo, [&quad,&bzr](const uvector<real, N> &_x, const real _w)
            {
                if (bernstein::evalBernsteinPoly(bzr, _x) < 0)
                {
                    quad.points.push_back(_x);
                    quad.weights.push_back(_w);
                }
            });

            if (_iso_bounds)
            {
                createQuadratureIsoBounds(bzr, _qo, *elem_quad);
            }
        }
        else
        {
            const uvector<real,N> midpt(0.5);
            elem_quad->interior = bernstein::evalBernsteinPoly(bzr, midpt) < 0;
        }


        return elem_quad;
    }

    template<int N>
    std::shared_ptr<const quad::ElementQuadrature<N>>
    createQuadrature(const BezierTP<N, 1> &_bezier_0,
                     const BezierTP<N, 1> &_bezier_1,
                     const int _qo,
                     const bool _iso_bounds)
    {
        const auto elem_quad = std::make_shared<quad::ElementQuadrature<N>>();

        auto &bnd_quad_0 = elem_quad->bdry_quads["levelset_0"];
        auto &bnd_quad_1 = elem_quad->bdry_quads["levelset_1"];

        const auto &bzr_0 = _bezier_0.getXarray();
        const auto &bzr_1 = _bezier_1.getXarray();

        ImplicitPolyQuadrature<N> ipquad(bzr_0, bzr_1);

        ipquad.integrate_surf(AlwaysGL, _qo, [&bnd_quad_0,&bnd_quad_1,&bzr_0,&bzr_1](const uvector<real,N> &_x, const real _w, const uvector<real,N>& _wn)
        {
            const int i = std::abs(bernstein::evalBernsteinPoly(bzr_0, _x)) < std::abs(bernstein::evalBernsteinPoly(bzr_1, _x)) ? 0 : 1;

            auto &bnd_quad = i == 0 ? bnd_quad_0 : bnd_quad_1;
            bnd_quad.points.push_back(_x);
            bnd_quad.weights.push_back(_w);

            // bnd_quad.normals.push_back(_wn);

            // TODO: I have to study better which is the output _wn.
            // I don't understand the difference between the
            // single and aggregated cases and it's crucial for this quantity.

            const auto &bzr = i == 0 ? bzr_0 : bzr_1;
            uvector<real,N> n = bernstein::evalBernsteinPolyGradient(bzr, _x);

            if (norm(n) > 0)
                n *= real(1.0) / norm(n);
            bnd_quad.normals.push_back(n);

        });

        elem_quad->cut = !bnd_quad_0.points.empty() || !bnd_quad_1.points.empty();

        if (elem_quad->cut)
        {
            auto &quad = elem_quad->quad;
            elem_quad->interior = false;
            ipquad.integrate(AlwaysGL, _qo, [&quad,&bzr_0,&bzr_1](const uvector<real, N> &_x, const real _w)
            {
                if (bernstein::evalBernsteinPoly(bzr_0, _x) < 0 && bernstein::evalBernsteinPoly(bzr_1, _x) < 0 )
                {
                    quad.points.push_back(_x);
                    quad.weights.push_back(_w);
                }
            });

            if (_iso_bounds)
            {
                createQuadratureIsoBounds(bzr_0, bzr_1, _qo, *elem_quad);
            }
        }
        else
        {
            const uvector<real,N> midpt(0.5);
            elem_quad->interior = bernstein::evalBernsteinPoly(bzr_0, midpt) < 0 && bernstein::evalBernsteinPoly(bzr_1, midpt) < 0;
        }


        return elem_quad;
    }

} // namespace algoim::bezier

#endif // ALGOIM_QUADRATURE_BEZIER_H
