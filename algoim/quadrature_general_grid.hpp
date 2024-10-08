#ifndef ALGOIM_QUADRATURE_GENERAL_GRID_H
#define ALGOIM_QUADRATURE_GENERAL_GRID_H

#include "quadrature_general.hpp"
#include "quadrature_element.hpp"
#include "quadrature_general_decomposition.hpp"

#include <map>

namespace algoim
{
    template<int N>
    struct Quadrature
    {
        std::vector<int> int_elems;
        std::vector<int> ext_elems;
        std::map<int, std::shared_ptr<const quad::ElementQuadrature<N>>> cut_elems;
    };

    namespace detail
    {
        inline bool compare_values(const real approx, const real target)
        {
            constexpr real abs_tol = 10.0 * std::numeric_limits<real>::epsilon();
            constexpr real rel_tol = 1e3 * abs_tol;

            return std::fabs(approx - target) <= (abs_tol + rel_tol * std::fabs(target));
        }

        template<int N, typename F>
        void
        createQuadratureIsoBounds(const F& _phi,
                                  const HyperRectangle<real,N>& domain,
                                  const int _qo,
                                  quad::ElementQuadrature<N> &_elem_quad)
        {
            for(int dir = 0, iso_bdry_id = 0; dir < N; ++dir)
            {
                for(int side = 0; side < 2; ++dir, ++iso_bdry_id)
                {
                    const auto face_quad = quadGen(_phi, domain, dir, side, _qo);

                    auto &iso_bdry_quad = _elem_quad.iso_bdry_quad[iso_bdry_id];

                    if (face_quad.nodes.empty())
                    {
                        iso_bdry_quad.cut = false;
                        iso_bdry_quad.interior = false;
                    }
                    else
                    {
                        const real elem_face_srf = face_quad.sumWeights();
                        const real domain_face_srf = prod(remove_component(domain.extent(), dir));
                        iso_bdry_quad.cut = !detail::compare_values(elem_face_srf, domain_face_srf);
                        iso_bdry_quad.interior = true;

                        if (iso_bdry_quad.cut)
                        {
                            iso_bdry_quad.fillIso(face_quad.nodes, iso_bdry_id);
                        }
                    }
                }
            }
        }

        template<int N>
        inline bool check_full_element(const QuadratureRule<N> &_quad, const HyperRectangle<real,N>& _domain)
        {
            const real elem_vol = _quad.sumWeights();
            const real domain_vol = prod(_domain.extent());
            return detail::compare_values(elem_vol, domain_vol);
        }
    }

    template<int N, typename F>
    std::shared_ptr<const Quadrature<N>>
    createQuadrature(const F& _phi,
                     const Grid<N> &_grid,
                     const int _qo,
                     const bool _iso_bounds)
    {
        const auto domain_quad = std::make_shared<Quadrature<N>>();

        auto &int_elems = domain_quad->int_elems;
        auto &ext_elems = domain_quad->ext_elems;
        auto &cut_elems = domain_quad->cut_elems;
        std::vector<int> unknown_elems;

        general::decompose<N,F>(_phi, SubGrid<N>(_grid), int_elems, ext_elems, unknown_elems);

        for(const auto &elem_id : unknown_elems)
        {
            const auto domain = _grid.getDomain(elem_id);

            const auto srf_quad = quadGen(_phi, domain, -1, N, _qo);
            const auto vol_quad = quadGen(_phi, domain, -1, 0, _qo);

            if (srf_quad.nodes.empty())
            {
                if (vol_quad.sumWeights() > 0.0)
                    int_elems.push_back(elem_id);
                else
                    ext_elems.push_back(elem_id);
                continue;
            }

            const auto elem_quad = std::make_shared<quad::ElementQuadrature<N>>();

            elem_quad->cut = true;
            elem_quad->quad.fill(vol_quad.nodes);

            auto &bnd_quad = elem_quad->bdry_quads["levelset"];
            bnd_quad.fill(srf_quad.nodes, _phi);

            if (_iso_bounds)
                detail::createQuadratureIsoBounds(_phi, domain, _qo, *elem_quad);

            cut_elems.emplace(elem_id, elem_quad);
        }

        std::sort(int_elems.begin(), int_elems.end());
        std::sort(ext_elems.begin(), ext_elems.end());

        return domain_quad;
    }

} // namespace algoim

#endif
