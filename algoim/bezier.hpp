#ifndef ALGOIM_BEZIER_H
#define ALGOIM_BEZIER_H

#include "change_basis.hpp"
#include "hyperrectangle.hpp"
#include "real.hpp"
#include "uvector.hpp"
#include "multiloop.hpp"
#include "polynomial_tp.hpp"
#include "binomial.hpp"
#include "xarray.hpp"
#include "lagrange.hpp"

#include <array>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>
#include <map>

namespace algoim::bezier
{

 /**
  * @brief Evaluates the Bernstein polynomials of the given order.
  * 
  * @tparam T_ Type of the input coordinate.
  * @tparam V Type of the output vector of basis values.
  * @param _x Evaluation point.
  * @param _order Order of the polynomial.
  * @param _values Computed basis values.
  */
 template<typename T_, typename V>
 static void evaluateBernsteinVal(const T_ &_x, const int _order, V &_values)
 {
     assert(0 < _order);

     const real* binom = Binomial::row(_order - 1);

     const T_ one(1.0);

     T_ p = one;
     for (int i = 0; i < _order; ++i)
     {
         _values[i] = p * binom[i];
         p *= _x;
     }
     p = 1.0;
     for (int i = _order - 1; i >= 0; --i)
     {
         _values[i] *= p;
         p *= one - _x;
     }
 }

 /**
  * @brief Evaluates the Bernstein polynomials of the given order (or its derivative).
  * 
  * @tparam T_ Type of the input coordinate.
  * @tparam V Type of the output vector of basis values.
  * @param _x Evaluation point.
  * @param _order Order of the polynomial.
  * @param _der Order of the derivative to be computed.
  * If 0, the value itself is computed.
  * @param _values Computed basis values.
  */
 template<typename T_, typename V>
 static void evaluateBernstein(const T_ &_x, const int _order, int _der, V &_values)
 {
     assert(0 <= _der);
     assert(0 < _order);

     if (_der == 0)
     {
         evaluateBernsteinVal(_x, _order, _values);
     }
     else if (_order <= _der)
     {
         for(int i = 0; i < _order; ++i)
             _values[i] = 0.0;
     }
     else
     {
         evaluateBernstein(_x, _order-1, _der - 1, _values);

         const int degree = _order - 1;

         _values[degree] = degree * _values[degree-1];
         for(int i = degree-1; i > 0; --i)
             _values[i] = degree * (_values[i-1] - _values[i]);
         _values[0] = -degree * _values[0];
     }
 }

/**
 * @brief N-dimensional tensor-product Bezier polynomial function.
 * 
 * @tparam T Base type of the coefficient variables.
 * @tparam N Dimension of the parametric domain.
 * @tparam R Dimension of the image.
 */
template<int N, int R = 1, typename T = real>
struct BezierTP : public PolynomialTP<N, R, T>
{
    /// Parent type.
    using Parent = PolynomialTP<N, R, T>;

    /// Coefs type.
    using CoefsType = typename Parent::CoefsType;

    /**
     * @brief Construct a new Bezier object.
     * 
     * The evaluation domain is assumed to be [0,1]^N.
     * 
     * @param _coefs Coefficients of the polynomial.
     * The ordering is such that dimension N-1 is inner-most, i.e.,
     * iterates the fastest, while dimension 0 is outer-most and iterates the slowest.
     * @param _order Order of the polynomial along the N dimensions.
     *               All the orders must be greater than zero.
     */
    BezierTP(const std::vector<CoefsType> &_coefs,
             const uvector<int, N> &_order)
             :
             Parent(_coefs, _order),
             coefs_xarray(this->coefs.data(), this->order)
    {}

    /**
     * @brief Construct a new Bezier object.
     * 
     * The evaluation domain is assumed to be [0,1]^N.
     * 
     * Coefficients vector is allocated.
     * 
     * @param _order Order of the polynomial along the N dimensions.
     *               All the orders must be greater than zero.
     */
    BezierTP(const uvector<int, N> &_order)
             :
             Parent(_order),
             coefs_xarray(this->coefs.data(), this->order)

    {}

    /**
     * @brief Creator.
     * 
     * @param _coefs Values of the spline control points.
     * The ordering is such that dimension N-1 is inner-most, i.e.,
     * iterates the fastest, while dimension 0 is outer-most and iterates the slowest.
     * @param _order Order of the polynomial along the N dimensions.
     *               All the orders must be greater than zero.
     * @return Created Bezier wrapped in a shared pointer.
     *
     * @note This static function exists for being called from the Python wrapper.
     */
    static std::shared_ptr<BezierTP>
    create(const std::vector<CoefsType> &_coefs,
           const std::array<int, N> &_order)
    {
        uvector<int, N> order;
        for(int dir = 0; dir < N; ++dir)
            order(dir) = _order[dir];

        return std::make_shared<BezierTP>(_coefs, order);
    }

    /**
     * @brief Construct a new Bezier whose image corresponds to the given one.
     * 
     * @param _image Image of the created Bezier.
     * @param _order Order of the created Bezier.
     */
    template<int R_ = R,
             std::enable_if_t<R_ == N && R_ == R, bool> = true>
    BezierTP(const HyperRectangle<real, N> &_image,
             const uvector<int, N> &_order)
             :
             BezierTP(_order)
    {
        if constexpr (N == 1)
        {
            this->coefs[0] = _image.range(0)(0);
            this->coefs[1] = _image.range(1)(0);

            int order = 2;
            while (order < _order(0))
            {
                this->coefs[order] = _image.range(1)(0);
                for(int i = order-1; i > 0; --i)
                    this->coefs[i] = real(i) / real(order + 1) * this->coefs[i-1] + (order + 1 - i) / real(order + 1) * this->coefs[i];

                ++order;
            }
        }
        else // if constexpr (1 < N)
        {
            std::vector<std::shared_ptr<BezierTP<1,1>>> beziers_1D;
            beziers_1D.reserve(N);
            for(int dir = 0; dir < N; ++dir)
            {
                const HyperRectangle<real, 1> domain_1D(_image.range(0)(dir), _image.range(1)(dir));
                const uvector<int,1> order_1D(_order(dir));
                beziers_1D.push_back(std::make_shared<BezierTP<1,1>>(domain_1D, order_1D));
            }

            auto it = this->coefs.begin();
            for (MultiLoop<N> i(0, _order); ~i; ++i)
            {
                auto &pt = *it++;
                for(int dir = 0; dir < N; ++dir)
                    pt(dir) = beziers_1D[dir]->coefs[i(dir)];
            }
        }
    }

    /**
     * @brief Construct a new Bezier whose image corresponds to the given one.
     * 
     * @param _image Image of the created Bezier.
     * @param _order Order of the created Bezier as std::array.
     */
    template<int R_ = R,
             std::enable_if_t<R_ == N && R_ == R, bool> = true>
    BezierTP(const HyperRectangle<real, N> &_image,
             const std::array<int, N> &_order)
             :
             BezierTP(_image, util::toUvector(_order))
    {}

    /**
     * @brief Creates a copy and returns it wrapped in a shared pointer.
     * 
     * @return Copy wrapped in a shared pointer.
     */
    std::shared_ptr<BezierTP> clone() const
    {
        return std::make_shared<BezierTP>(this->coefs, this->order);
    }

    /**
     * @brief Creates a Bezier from a Lagrange element.
     * 
     * @param _lagr Lagrange element to be transformed.
     * @return Created Bezier.
     */
    static std::shared_ptr<BezierTP<N,R,T>>
    create(const lagrange::LagrangeTP<N, R, T> &_lagr)
    {
        const auto &order = _lagr.order;
        const auto bzr = std::make_shared<BezierTP<N,R,T>>(order);

        change_basis::changeLagrangeToBezier(order, _lagr.coefs.data(), bzr->coefs.data());

        return bzr;
    }

    /**
     * @brief Gets the dimension of the Bezier.
     * @return Dimension of the Bezier.
     */
    static int getDim()
    {
        return N;
    }

    /**
     * @brief Gets the range of the Bezier.
     * @return Range of the Bezier.
     */
    static int getRange()
    {
        return R;
    }

    /**
     * @brief Changes the sign of the coefficients.
     */
    void negate()
    {
        for(auto &c : this->coefs)
            c *= -1.0;
    }

    /**
     * @brief Adds a constant value to the coefficients.
     * @param _val Value to be added.
     */
    void add(const T _val)
    {
        for(auto &c : this->coefs)
            c += _val;
    }


private:
    /// dim-dimensional array view of the coefficients.
    xarray<CoefsType, N> coefs_xarray;

public:
    /**
     * @brief Evaluates the polynomial function at a given point.
     * 
     * @tparam T_ Type of the input and output variables.
     * @param _x Evaluation point.
     * @return Value of the polynomial at @p _x.
     */
    template<typename T_>
    typename Parent::template Val<T_> operator() (const uvector<T_, N>& _x) const
    {
        auto coefs_it = this->coefs.cbegin();
        return casteljau<T_, N>(_x, coefs_it, this->order);
    }

    /**
     * @brief Evaluates the polynomial gradient at a given point.
     * 
     * @tparam T_ Type of the input and output variables.
     * @param _x Evaluation point.
     * @return Gradient of the polynomial at @p _x.
     */
    template<typename T_>
    typename Parent::template Grad<T_> grad(const uvector<T_, N>& _x) const
    {
        auto coefs_it = this->coefs.cbegin();
        const auto ref_grad = remove_component(casteljauDer<T_, N>(_x, coefs_it, this->order), 0);
        return ref_grad;
    }

    /**
     * @brief Evaluates the polynomial Hessian at a given point.
     * 
     * @tparam T_ Type of the input and output variables.
     * @param _x Evaluation point.
     * @return Hessian of the polynomial at @p _x.
     */
    template<typename T_>
    typename Parent::template Hess<T_> hessian(const uvector<T_, N>& _x) const
    {
        using C = std::conditional_t<R == 1, T_, uvector<T_, R>>;

        // TODO: in the future this may be improved by implementing the Casteljau algorithm for second derivatives.
        std::array<std::array<std::vector<T_>, N>, 3> basis_and_ders;
        for(int dir = 0; dir < N; ++dir)
        {
            const auto o = this->order(dir);
            for(int der = 0; der <= 2; ++der)
            {
                basis_and_ders[der][dir].resize(o);
                evaluateBernstein(_x(dir), o, der, basis_and_ders[der][dir]);
            }
        }

        std::array<int, N> ders;
        for(int dir = 0; dir < N; ++dir)
            ders[dir] = 0;

        typename Parent::template Hess<T_> h(0.0);
        for(int i = 0, k = 0; i < N; ++i)
        {
            ++ders[i];
            for(int j = i; j < N; ++j, ++k)
            {
                ++ders[j];

                auto &h_ = h(k);
                h_ = 0.0;

                auto coefs_it = this->coefs.cbegin();
                for (MultiLoop<N> l(0, this->order); ~l; ++l, ++coefs_it)
                {
                    C v{*coefs_it};
                    for(int dir = 0; dir < N; ++dir)
                        v *= basis_and_ders[ders[dir]][dir][l(dir)];
                    h_ += v;
                }

                --ders[j];
            }
            --ders[i];
        }

        return  h;
    }

    /**
     * @brief Gets a constant reference to the stored xarray view of the polynomial coefficients.
     * @return Constant view of the polynomial coefficients.
     */
    const xarray<CoefsType, N> &getXarray() const
    {
        return coefs_xarray;
    }

private:

    /**
     * @brief Evaluates a Bezier polynomial using the Casteljau's algorithm.
     * 
     * For the algorithm details check:
     *   https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
     * 
     * The evaluation along the dimensions is performed by recursively calling this function.
     * 
     * @tparam T_ Type of the input and output variables.
     * @tparam M Recursive dimension of the polynomial.
     * @param _x Evaluation point.
     * @param _coefs Iterator to the coefficients of the polynomial.
     * Their ordering is the same as the coefficients members of the class.
     * @param _order Order of the polynomial along the M dimensions.
     * @return Value of the polynomial at @p _x.
     */
    template<typename T_, int M>
    static typename Parent::template Val<T_> casteljau(const uvector<T_,M>& _x,
                                                       typename std::vector<CoefsType>::const_iterator &_coefs,
                                                       const uvector<int,M> _order)
    {
        static_assert(M > 0, "Invalid dimension.");
        assert(0 < min(_order));

        const int order = _order(0);

        using V = typename Parent::template Val<T_>;
        std::vector<V> beta(order);
        if constexpr (M == 1)
        {
            beta.assign(_coefs, _coefs + order);
            _coefs += order;
        }
        else
        {
            for (int i = 0; i < order; ++i)
                beta[i] = casteljau<T_, M-1>(remove_component(_x, 0), _coefs, remove_component(_order, 0));
        }

        const V x = _x(0);
        const V one_x = 1.0 - x;
        for(int j = 1; j < order; ++j)
        {
            for(int k = 0; k < (order - j); ++k)
                beta[k] = beta[k] * one_x + beta[k + 1] * x;
        }

        return beta[0];
    }

    /**
     * @brief Evaluates the gradient Bezier polynomial using the Casteljau's algorithm.
     * 
     * For the algorithm details check:
     *   https://en.wikipedia.org/wiki/De_Casteljau%27s_algorithm
     * 
     * The evaluation along the dimensions is performed by recursively calling this function.
     * 
     * @tparam T_ Type of the input and output variables.
     * @tparam M Recursive dimension of the polynomial.
     * @param _x Evaluation point.
     * @param _coefs Iterator to the coefficients of the polynomial.
     * Their ordering is the same as the coefficients members of the class.
     * @param _order Order of the polynomial along the M dimensions.
     * @return The first component of the return vector corresponds to the value of the polynomial itself,
     * while the gradient is stored in the following M components.
     */
    template<typename T_, int M>
    static uvector<typename Parent::template Val<T_>, M+1> casteljauDer(const uvector<T_, M>& _x,
                                                                        typename std::vector<CoefsType>::const_iterator &_coefs,
                                                                        const uvector<int, M> _order)
    {
        static_assert(M > 0, "Invalid dimension.");
        assert(0 < min(_order));

        const int order = _order(0);
        const int degree = order - 1;

        using V = typename Parent::template Val<T_>;
        std::vector<uvector<V, M>> beta(order);
        if constexpr (M == 1)
        {
            for (int i = 0; i < order; ++i)
                beta[i](0) = *_coefs++;
        }
        else
        {
            for (int i = 0; i < order; ++i)
                beta[i] = casteljauDer<T_, M-1>(remove_component(_x, 0), _coefs, remove_component(_order, 0));
        }

        std::vector<V> gamma(order, V(0.0));
        gamma[0] += - degree * beta[0](0);
        gamma[1] += - beta[0](0);
        for(int i = 1; i < degree; ++i)
        {
            gamma[i-1] += (order - i) * beta[i](0);
            gamma[i] += (2 * i - degree) * beta[i](0);
            gamma[i+1] += - (i+1) * beta[i](0);
        }
        gamma[degree-1] += beta[degree](0);
        gamma[degree] += degree * beta[degree](0);

        const T_ x = _x(0);
        for(int j = 1; j < order; ++j)
        {
            for(int k = 0; k < (order - j); ++k)
            {
                beta[k] = beta[k] * (1 - x) + beta[k + 1] * x;
                gamma[k] = gamma[k] * (1 - x) + gamma[k + 1] * x;
            }
        }

        return add_component(beta[0], 1, gamma[0]);
    }

};

namespace detail
{

template<int R = 1, typename T = real>
std::shared_ptr<BezierTP<1,R,T>>
product(const BezierTP<1, R, T> &_lhs, const BezierTP<1, R, T> &_rhs)
{
    static const int N = 1;
    using C = typename BezierTP<N,R,T>::CoefsType;

    const auto prod_bzr = std::make_shared<BezierTP<N,R,T>>(_lhs.order + _rhs.order - 1);
    const int o0 = prod_bzr->order(0);

    auto &coefs         = prod_bzr->coefs;
    coefs.assign(coefs.size(), C(0.0));

    const auto compute_aux_vector = [](const auto &_coefs, const uvector<int, N> &_order, auto &_vec) {

        _vec.resize(prod(_order));
        auto it = _vec.begin();

        const real* binomials_u = Binomial::row(_order(0)-1);

        for (int i = 0; i < _order(0); ++i)
            *it++ = _coefs[i] * binomials_u[i];
    };

    std::vector<C> aux_lhs, aux_rhs;

    compute_aux_vector(_lhs.coefs, _lhs.order, aux_lhs);
    compute_aux_vector(_rhs.coefs, _rhs.order, aux_rhs);

    C *const c  = coefs.data();
    const C *a0 = aux_lhs.data();

    // Note: this is the most expensive part of the algorithm.
    // It is a full 3D linear convolution.
    // Probably it can be speed up using FFT algorithms
    // (or AVX extensions?).

    for (int i = 0; i < _lhs.order(0); ++i)
    {
        // Note: this is the most expensive part.
        const C c0  = *a0++;
        const C *a1 = aux_rhs.data();

        C *ptr = c + i;
        for (int l = 0; l < _rhs.order(0); ++l)
            *ptr++ += c0 * *a1++;
    }     // i

    const real* binomials_u = Binomial::row(o0-1);

    C *ptr = c;

    for (int i = 0; i < o0; ++i)
        *ptr++ /= binomials_u[i];

    return prod_bzr;
}

template<int R = 1, typename T = real>
std::shared_ptr<BezierTP<2,R,T>>
product(const BezierTP<2, R, T> &_lhs, const BezierTP<2, R, T> &_rhs)
{
    static const int N = 2;
    using C = typename BezierTP<N,R,T>::CoefsType;

    const auto prod_bzr = std::make_shared<BezierTP<N,R,T>>(_lhs.order + _rhs.order - 1);
    const int o0 = prod_bzr->order(0);
    const int o1 = prod_bzr->order(1);

    auto &coefs         = prod_bzr->coefs;
    coefs.assign(coefs.size(), C(0.0));

    const auto compute_aux_vector = [](const auto &_coefs, const uvector<int, N> &_order, auto &_vec) {

        _vec.resize(prod(_order));
        auto it = _vec.begin();

        const real* binomials_u = Binomial::row(_order(0)-1);
        const real* binomials_v = Binomial::row(_order(1)-1);

        // Lexicographical.
        // for (int j = 0, ij = 0; j < _order(1); ++j) {
        //     for (int i = 0; i < _order(0); ++i, ++ij)
        //        *it++ = _coefs[ij] * binomials_u[i] * binomials_v[j];
        // } // j

        for (int i = 0, ij = 0; i < _order(0); ++i)
        {
            for (int j = 0; j < _order(1); ++j, ++ij)
            {
                *it++ = _coefs[ij] * binomials_u[i] * binomials_v[j];
            } // j
        } // i
    };

    std::vector<C> aux_lhs, aux_rhs;

    compute_aux_vector(_lhs.coefs, _lhs.order, aux_lhs);
    compute_aux_vector(_rhs.coefs, _rhs.order, aux_rhs);

    C *const c  = coefs.data();
    const C *a0 = aux_lhs.data();

    // Note: this is the most expensive part of the algorithm.
    // It is a full 3D linear convolution.
    // Probably it can be speed up using FFT algorithms
    // (or AVX extensions?).

    // Lexicographical
    // for (int k = 0; k < _lhs.order(2); ++k) {
    //     for (int j = 0; j < _lhs.order(1); ++j) {
    //         for (int i = 0; i < _lhs.order(0); ++i) {

    //             // Note: this is the most expensive part.
    //             const C c0  = *a0++;
    //             const C *a1 = aux_rhs.data();
    //             for (int n = 0; n < _rhs.order(2); ++n) {
    //                 for (int m = 0; m < _rhs.order(1); ++m) {
    //                    C *ptr = c + (k + n) * o0 * o1 + (j + m) * o0 + i;
    //                    for (int l = 0; l < _rhs.order(0); ++l)
    //                        *ptr++ += c0 * *a1++;
    //                 } // m
    //             } // n

    //           // // Is this more stable? (but definitely slower ...)
    //           // // The divisions below must be commented.
    //           // for (int n = 0; n < lhs_degrees(2); ++n) {
    //           //   for (int m = 0; m < lhs_degrees(1); ++m) {
    //           //     const T_ c0_bn = c0 / bn_u[j + m] / bn_u[k + n];
    //           //     T_ *const ptr  = c + (k + n) * (prod_bzr.get_degrees()[1] + 1) * (prod_bzr.get_degrees()[0] + 1) +
    //           //                     (j + m) * (prod_bzr.degrees_[0] + 1) + i;
    //           //     for (int l = 0; l < lhs_degrees(0); ++l) {
    //           //       ptr[l] += c0_bn * *a1++ / bn_u[l + i];
    //           //     } // l
    //           //   }   // m
    //           // }     // n
    
    //         } // i
    //     }   // j
    // }     // k

    for (int i = 0; i < _lhs.order(0); ++i)
    {
        for (int j = 0; j < _lhs.order(1); ++j)
        {
            // Note: this is the most expensive part.
            const C c0  = *a0++;
            const C *a1 = aux_rhs.data();

            for (int l = 0; l < _rhs.order(0); ++l) {
                C *ptr = c + (i + l) * o1 + j;
                for (int m = 0; m < _rhs.order(1); ++m) {
                    *ptr++ += c0 * *a1++;
                } // m
            } // l
        }   // j
    }     // i

    const real* binomials_u = Binomial::row(o0-1);
    const real* binomials_v = Binomial::row(o1-1);

    C *ptr = c;

    // // Lexicographical
    // for (int j = 0; j < o1; ++j) {
    //     for (int i = 0; i < o0; ++i)
    //          *ptr++ /= (binomials_u[i] * binomials_v[j]);
    // } // j

    for (int i = 0; i < o0; ++i) {
        for (int j = 0; j < o1; ++j) {
            *ptr++ /= (binomials_u[i] * binomials_v[j]);
        } // j
    }   // i

    return prod_bzr;
}

template<int R = 1, typename T = real>
std::shared_ptr<BezierTP<3,R,T>>
product(const BezierTP<3, R, T> &_lhs, const BezierTP<3, R, T> &_rhs)
{
    static const int N = 3;
    using C = typename BezierTP<N,R,T>::CoefsType;

    const auto prod_bzr = std::make_shared<BezierTP<N,R,T>>(_lhs.order + _rhs.order - 1);
    const int o0 = prod_bzr->order(0);
    const int o1 = prod_bzr->order(1);
    const int o2 = prod_bzr->order(2);

    auto &coefs         = prod_bzr->coefs;
    coefs.assign(coefs.size(), C(0.0));

    const auto compute_aux_vector = [](const auto &_coefs, const uvector<int, N> &_order, auto &_vec) {

        _vec.resize(prod(_order));
        auto it = _vec.begin();

        const real* binomials_u = Binomial::row(_order(0)-1);
        const real* binomials_v = Binomial::row(_order(1)-1);
        const real* binomials_w = Binomial::row(_order(2)-1);

        // Lexicographical.
        // for (int k = 0, kji = 0; k < _order(2); ++k) {
        //     for (int j = 0; j < _order(1); ++j) {
        //         for (int i = 0; i < _order(0); ++i, ++kji)
        //            *it++ = _coefs[ijk] * binomials_u[i] * binomials_v[j] * binomials_w[k];
        //     } // j
        // }   // k

        for (int i = 0, ijk = 0; i < _order(0); ++i)
        {
            for (int j = 0; j < _order(1); ++j)
            {
                for (int k = 0; k < _order(2); ++k, ++ijk)
                    *it++ = _coefs[ijk] * binomials_u[i] * binomials_v[j] * binomials_w[k];
            } // j
        } // i
    };

    std::vector<C> aux_lhs, aux_rhs;

    compute_aux_vector(_lhs.coefs, _lhs.order, aux_lhs);
    compute_aux_vector(_rhs.coefs, _rhs.order, aux_rhs);

    C *const c  = coefs.data();
    const C *a0 = aux_lhs.data();

    // Note: this is the most expensive part of the algorithm.
    // It is a full 3D linear convolution.
    // Probably it can be speed up using FFT algorithms
    // (or AVX extensions?).

    // Lexicographical
    // for (int k = 0; k < _lhs.order(2); ++k) {
    //     for (int j = 0; j < _lhs.order(1); ++j) {
    //         for (int i = 0; i < _lhs.order(0); ++i) {

    //             // Note: this is the most expensive part.
    //             const C c0  = *a0++;
    //             const C *a1 = aux_rhs.data();
    //             for (int n = 0; n < _rhs.order(2); ++n) {
    //                 for (int m = 0; m < _rhs.order(1); ++m) {
    //                    C *ptr = c + (k + n) * o0 * o1 + (j + m) * o0 + i;
    //                    for (int l = 0; l < _rhs.order(0); ++l)
    //                        *ptr++ += c0 * *a1++;
    //                 } // m
    //             } // n

    //           // // Is this more stable? (but definitely slower ...)
    //           // // The divisions below must be commented.
    //           // for (int n = 0; n < lhs_degrees(2); ++n) {
    //           //   for (int m = 0; m < lhs_degrees(1); ++m) {
    //           //     const T_ c0_bn = c0 / bn_u[j + m] / bn_u[k + n];
    //           //     T_ *const ptr  = c + (k + n) * (prod_bzr.get_degrees()[1] + 1) * (prod_bzr.get_degrees()[0] + 1) +
    //           //                     (j + m) * (prod_bzr.degrees_[0] + 1) + i;
    //           //     for (int l = 0; l < lhs_degrees(0); ++l) {
    //           //       ptr[l] += c0_bn * *a1++ / bn_u[l + i];
    //           //     } // l
    //           //   }   // m
    //           // }     // n
    
    //         } // i
    //     }   // j
    // }     // k

    for (int i = 0; i < _lhs.order(0); ++i)
    {
        for (int j = 0; j < _lhs.order(1); ++j)
        {
            for (int k = 0; k < _lhs.order(2); ++k)
            {

                // Note: this is the most expensive part.
                const C c0  = *a0++;
                const C *a1 = aux_rhs.data();

                for (int l = 0; l < _rhs.order(0); ++l) {
                    for (int m = 0; m < _rhs.order(1); ++m) {
                        C *ptr = c + (i + l) * o2 * o1 + (j + m) * o2 + k;
                        for (int n = 0; n < _rhs.order(2); ++n)
                            *ptr++ += c0 * *a1++;
                    } // m
                } // l
    
            } // k
        }   // j
    }     // i

    const real* binomials_u = Binomial::row(o0-1);
    const real* binomials_v = Binomial::row(o1-1);
    const real* binomials_w = Binomial::row(o2-1);

    C *ptr = c;

    // // Lexicographical
    // for (int k = 0; k < o2; ++k) {
    //     for (int j = 0; j < o1; ++j) {
    //         for (int i = 0; i < o0; ++i)
    //             *ptr++ /= (binomials_u[i] * binomials_v[j] * binomials_w[k]);
    //     } // j
    // }   // k

    for (int i = 0; i < o0; ++i) {
        for (int j = 0; j < o1; ++j) {
            for (int k = 0; k < o2; ++k)
                *ptr++ /= (binomials_u[i] * binomials_v[j] * binomials_w[k]);
        } // j
    }   // i

    return prod_bzr;
}

} // namespace detail

template<int N, int R = 1, typename T = real>
std::shared_ptr<BezierTP<N,R,T>>
product(const BezierTP<N, R, T> &_lhs, const BezierTP<N, R, T> &_rhs)
{
    return detail::product(_lhs, _rhs);
}


} // namespace algoim::bezier


#endif // ALGOIM_BEZIER_H
