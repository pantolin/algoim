#ifndef ALGOIM_LAGRANGE_TP_H
#define ALGOIM_LAGRANGE_TP_H

/* Tensor-product Lagrange elements. */

#include "hyperrectangle.hpp"
#include "real.hpp"
#include "utility.hpp"
#include "uvector.hpp"
#include "multiloop.hpp"
#include "polynomial_tp.hpp"

#include <vector>
#include <cassert>

namespace algoim::lagrange
{
    /**
     * @brief N-dimensional tensor-product Lagrance polynomial function.
     * 
     * @tparam T Base type of the coefficient variables.
     * @tparam N Dimension of the parametric domain.
     * @tparam R Dimension of the image.
     */
    template<int N, int R, typename T = real>
    struct LagrangeTP : public PolynomialTP<N, R, T>
    {
        /// Parent type.
        using Parent = PolynomialTP<N, R, T>;

        /// Coefs type.
        using CoefsType = typename Parent::CoefsType;

        /**
         * @brief Construct a new Lagrange object.
         * 
         * The evaluation domain is assumed to be [0,1]^N.
         * 
         * @param _coefs Coefficients of the polynomial.
         * The ordering is such that dimension N-1 is inner-most, i.e.,
         * iterates the fastest, while dimension 0 is outer-most and iterates the slowest.
         * @param _order Order of the polynomial along the N dimensions.
         *               All the orders must be greater than zero.
         */
        LagrangeTP(const std::vector<CoefsType> &_coefs,
                   const uvector<int, N> &_order)
                  :
                  Parent(_coefs, _order)
        {}

        /**
         * @brief Construct a new Lagrange object.
         * 
         * The evaluation domain is assumed to be [0,1]^N.
         * 
         * Coefficients vector is allocated.
         * 
         * @param _order Order of the polynomial along the N dimensions.
         *               All the orders must be greater than zero.
         */
        LagrangeTP(const uvector<int, N> &_order)
                   :
                   Parent(_order)
        {}

        /**
         * @brief Construct a new Lagrange whose image corresponds to the given one.
         * 
         * @param _image Image of the created Lagrange.
         * @param _order Order of the created Lagrange.
         */
        LagrangeTP(const HyperRectangle<real, N> &_image,
                   const uvector<int, N> &_order)
                   :
                   LagrangeTP(_order)
        {
            auto it = this->coefs.begin();
            for (MultiLoop<N> i(0, _order); ~i; ++i)
            {
                auto &pt = *it++;
                for(int dir = 0; dir < N; ++dir)
                {
                    const real t = real(i(dir)) / real(_order(dir) - 1);
                    pt(dir) = (1.0 - t) * _image.range(0)(dir) + t * _image.range(1)(dir);
                }
            }
        }

        /**
         * @brief Construct a new Lagrange whose image corresponds to the given one.
         * 
         * @param _image Image of the created Lagrange.
         * @param _order Order of the created Lagrange.
         */
        LagrangeTP(const HyperRectangle<real, R> &_image,
                   const uvector<int, N> &_order,
                   const int _fix_dir,
                   const int _side)
                   :
                   LagrangeTP(_order)
        {
            static_assert((N + 1) == R, "Invalid dimensions.");
            assert(0 <= _fix_dir && _fix_dir < R);
            assert(_side == 0 || _side == 1);

            auto it = this->coefs.begin();
            for (MultiLoop<N> i(0, _order); ~i; ++i)
            {
                auto &pt = *it++;
                for(int dir = 0, dir2 = 0; dir < R; ++dir)
                {
                    if (dir == _fix_dir)
                    {
                        pt(dir) = _image.range(_side)(dir);
                    }
                    else
                    {
                        const real t = real(i(dir2)) / real(_order(dir2) - 1);
                        pt(dir) = (1.0 - t) * _image.range(0)(dir) + t * _image.range(1)(dir);
                        ++dir2;
                    }
                }
            }
        }

        /**
         * @brief Creates a copy and returns it wrapped in a shared pointer.
         * 
         * @return Copy wrapped in a shared pointer.
         */
        std::shared_ptr<LagrangeTP> clone() const
        {
            return std::make_shared<LagrangeTP>(this->coefs, this->order);
        }

        /**
         * @brief Extracts an (1D) edge of the element.
         * 
         * @param _edge_id Id of the edge to extract.
         * @return Extracted edge.
         */
        std::shared_ptr<LagrangeTP<1,R,T>> extractEdge(const int _edge_id) const
        {
            PolynomialTPEdgeIt<N> edge_it(this->order, _edge_id);

            const int order_1D = this->order(edge_it.act_dir);
            const auto edge = std::make_shared<LagrangeTP<1,R,T>>(uvector<int,1>(order_1D));

            for(int i = 0; ~edge_it; ++edge_it, ++i)
                edge->coefs[i] = this->coefs[util::toFlatIndex(this->order, edge_it())];

            return edge;
        }
    };

} // namespace algoim::lagrange

#endif // ALGOIM_LAGRANGE_TP_H