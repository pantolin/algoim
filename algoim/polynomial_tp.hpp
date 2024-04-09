#ifndef ALGOIM_POLYNOMIAL_TP_H
#define ALGOIM_POLYNOMIAL_TP_H

/* Tensor-product polynomial elements related functionalities. */


#include "real.hpp"
#include "uvector.hpp"
#include "hyperrectangle.hpp"
#include "utility.hpp"

#include <vector>
#include <cassert>

namespace algoim
{

    /**
     * @brief Base class for tensor-product polynomial functions.
     * 
     * @tparam N_ Dimension of the parametric domain.
     * @tparam R_ Dimension of the image.
     * @tparam T_ Base type of the coefficient variables.
     */
    template<int N_, int R_, typename T_ = real>
    struct PolynomialTP
    {
        /// Domain dimension.
        static const int N = N_;

        /// Image range.
        static const int R = R_;

        /// Variables type.
        using T = T_;

        /// Coefs type.
        using CoefsType = std::conditional_t<R == 1, T, uvector<T, R>>;

        /**
         * @brief Type of the value obtained with when the operator() is called.
         * @tparam TT Type of the input coordinates.
         */
        template<typename TT>
        using Val = std::conditional_t<R == 1, TT, uvector<TT, R>>;

        /**
         * @brief Type of the value obtained with when the grad() is called.
         * @tparam TT Type of the input coordinates.
         */
        template<typename TT>
        using Grad = uvector<Val<TT>, N>;

        /**
         * @brief Type of the value obtained with when the hess() is called (if implemented).
         * @tparam TT Type of the input coordinates.
         */
        template<typename TT>
        using Hess = uvector<Val<TT>, N*(N+1)/2>;

        /**
         * @brief Constructor.
         * 
         * The coefficients vector is allocated by the constructor.
         * 
         * @param _order Order of the polynomial along each parametric direction.
         */
        PolynomialTP(const uvector<int, N> _order)
        :
        order(_order),
        coefs(prod(_order))
        {
            static_assert(0 < N, "Dimension must be > 0.");
            assert(0 < min(order));
            assert(static_cast<std::size_t>(prod(order)) == coefs.size());
        }

        /**
         * @brief Constructor.
         * 
         * @param _coefs Coefficients of the polynomial.
         * The ordering is such that dimension N-1 is inner-most, i.e.,
         * iterates the fastest, while dimension 0 is outer-most and iterates the slowest.
         * @param _order Order of the polynomial along each parametric direction.
         */
        PolynomialTP(const std::vector<CoefsType> &_coefs, const uvector<int, N> _order)
        :
        order(_order),
        coefs(_coefs)
        {
            static_assert(0 < N, "Dimension must be > 0.");
            assert(0 < min(order));
            assert(static_cast<std::size_t>(prod(order)) == coefs.size());
        }

        /// Order of the polynomial along each parametric direction.
        uvector<int, N> order;
        /**
         * Coefficients of the polynomial.
         * The ordering is such that dimension N-1 is inner-most, i.e.,
         * iterates the fastest, while dimension 0 is outer-most and iterates the slowest.
         */
        std::vector<CoefsType> coefs;

        /**
         * @brief Get the number of coeficients.
         * 
         * @return Number of polynomial coefficients.
         */
        int getNumCoefs() const
        {
            return prod(order);
        }

        /**
         * @brief Transform a tensor index into a flat index.
         * 
         * @param _tid Tensor index to be transformed.
         * @return Computed flat index.
         */
        int toFlatIndex(const uvector<int,N_> &_tid) const
        {
            return util::toFlatIndex(this->order, _tid);
        }

        /**
         * @brief Transform a flat index into a tensor index.
         * 
         * @param _id Flat index to be transformed.
         * @return Computed tensor index.
         */
        uvector<int,N_> toTensorIndex(const int _id) const
        {
            return util::toTensorIndex(this->order, _id);
        }

        /**
         * @brief Transform (in place) the coefficients of the polynomial
         * from @p _old_domain to @p _new_domain.
         * 
         * @param _old_domain Domain from which polynomial coefficients are transformed from.
         * @param _new_domain Domain to which polynomial coefficients are transformed to.
         */
        void transformImage(const HyperRectangle<real, R> &_old_domain,
                             const HyperRectangle<real, R> &_new_domain)
        {
            for(auto &pt : coefs)
            {
                for(int dir = 0; dir < R; ++dir)
                {
                    pt(dir) = _new_domain.min(dir) + (pt(dir) - _old_domain.min(dir)) * _new_domain.extent(dir) / _old_domain.extent(dir);
                }
            }
        }
    };

    /**
     * @brief Iterator for edges of tensor product elements.
     * 
     * @tparam N Dimension of the parametric domain.
     */
    template<int N>
    struct PolynomialTPEdgeIt
    {
        /**
         * @brief Constructor.
         * 
         * @param _order Order of the element along all the parametric directions.
         * @param _edge_id Id of the element's edge (following lexicographical convention).
         */
        PolynomialTPEdgeIt(const uvector<int, N> &_order, const int _edge_id)
        : act_dir(getActiveDir(_edge_id)), order(_order), i(), min(), max(), valid(true)
        {
            this->setMinMax(_order, _edge_id);
        }

        /**
         * @brief Increments the iterator position.
         */
        PolynomialTPEdgeIt& operator++()
        {
            if (++i(act_dir) < max(act_dir))
                return *this;
            valid = false;
            return *this;
        }

        /**
         * @brief Returns a reference to the current tensor index.
         * 
         * @return Current tensor index.
         */
        const uvector<int,N>& operator()() const
        {
            return i;
        }

        /**
         * @brief Gets the flat index of the current iterator's position.
         * @return Flat index of the position.
         */
        int getFlatIndex() const
        {
            assert(valid && "Iterator is not in a valid state");
            return util::toFlatIndex(this->order, i);
        }

        /**
         * @brief Checks whether or not the iteration is valid
         * of not (reached the end).
         * 
         * @return True if the iterator is in a valid state, false otherwise.
         */
        bool operator~() const
        {
            return valid;
        }

        /**
         * @brief Resets the iterator to a valid state setting its tensor
         * index to the beginning of the edge.
         */
        void reset()
        {
            this->i(act_dir) = 0;
            this->valid = true;
        }

        /// Active parametric direction of the edge.
        const int act_dir;
private:
        /// Order of the element along all the parametric directions.
        const uvector<int,N> order;
        /// Tensor index of the iterator.
        uvector<int,N> i;
        /// Minimum and maximum tensor dimensions of the edge.
        uvector<int,N> min, max;
        /// Flag indicating if the iterator is in a valid state.
        bool valid;


        /**
         * @brief Sets the minimum and maximum tensor dimensions of the edge.
         * 
         * @param _order Polynomial order of the element along the parametric directions.
         * @param _edge_id If of the edge.
         */
        void setMinMax(const uvector<int, N> &_order, const int _edge_id)
        {
            if constexpr (N == 2)
            {
                if (_edge_id < 2)
                {
                    min(0) = _edge_id == 0 ? 0 : (_order(0) - 1);
                }
                else
                {
                    min(1) = _edge_id == 2 ? 0 : (_order(1) - 1);
                }
            }
            else // if constexpr (N == 3)
            {
                if (_edge_id < 4)
                {
                    min(0) = (_edge_id % 2) == 0 ? 0 : (_order(0) - 1);
                    min(1) = (_edge_id / 2) == 0 ? 0 : (_order(1) - 1);
                }
                else if (_edge_id < 8)
                {
                    min(0) = ((_edge_id - 4) % 2) == 0 ? 0 : (_order(0) - 1);
                    min(2) = ((_edge_id - 4) / 2) == 0 ? 0 : (_order(2) - 1);
                }
                else 
                {
                    min(1) = ((_edge_id - 8) % 2) == 0 ? 0 : (_order(1) - 1);
                    min(2) = ((_edge_id - 8) / 2) == 0 ? 0 : (_order(2) - 1);
                }
            }
            this->max = this->min;

            this->min(act_dir) = 0;
            this->max(act_dir) = _order(act_dir);

            this->i = this->min;
        }

        /**
         * @brief Gets the number of edge in an element of dimension N.
         * @return Number of edge.
         */
        static int getNumEdges()
        {
            static_assert(N == 2 || N == 3, "Not implemented.");
            return N == 2 ? 4 : 12;
        }

        /**
         * @brief Gets the active parametric direction of an edge.
         * 
         * @param _edge_id If of the edge.
         * @return Active parametric direction index.
         */
        static int getActiveDir(const int _edge_id)
        {
            static_assert(N == 2 || N == 3, "Not implemented.");
            assert(0 <= _edge_id && _edge_id < getNumEdges());

            int act_dir{-1};
            if constexpr (N == 2)
            {
                if (_edge_id < 2)
                    return 1;
                else
                    return 0;
            }
            else // if constexpr (N == 3)
            {
                if (_edge_id < 4)
                    return 2;
                else if (_edge_id < 8)
                    return 1;
                else 
                    return 0;
            }
        }
    };



} // namespace algoim

#endif // ALGOIM_POLYNOMIAL_TP_H
