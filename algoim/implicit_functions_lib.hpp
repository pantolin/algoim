#ifndef ALGOIM_IMPLICIT_FUNCTIONS_LIB_H
#define ALGOIM_IMPLICIT_FUNCTIONS_LIB_H

// A few implicit functions ready to be consumed by algoim

#include "uvector.hpp"
#include "real.hpp"
#include "utility.hpp"

#include <cassert>
#include <cmath>
#include <vector>
#include <numeric>

namespace algoim
{

namespace detail
{
    /**
     * @brief Transform a 2D or 3D value to a 3D one.
     * 
     * @tparam T Input and output's type.
     * @tparam N Input's dimension.
     * @param _a Value to transformed.
     * @return 
     */
    template<typename T, int N>
    static uvector<T, 3> to_3D(const uvector<T, N> &_a)
    {
      if constexpr (N == 2)
          return add_component(_a, 2, T(0.0));
      else // if constexpr (N == 3)
          return _a;
    }

    template<class T>
    inline constexpr T pow(const T _base, unsigned const _exponent)
    {
        return (_exponent == 0) ? 1 : (_base * pow(_base, _exponent-1));
    }

} // detail


/**
 * @brief Dimension independent spherical function.
 * The function is defined by a center and a radius.
 * The function presents a negative sign around the center,
 * and positive far away. At the a radius distance from the
 * center, the function vanishes.
 * 
 * @tparam N_ Parametric dimension.
 */
template<int N_>
struct Sphere
{
    /// Function's parametric dimension.
    static const int N = N_;

    /**
     * @brief Constructor.
     *
     * @param _center Sphere center.
     * @param _R Sphere radius.
     */
    Sphere(const uvector<real, N> &_center, const real _R)
    :
    center(_center), R(_R)
    {
        assert(0 <= _R && "Invalid radius.");
    }

    /**
     * @brief Constructor.
     *
     * @param _center Sphere center.
     * @param _R Sphere radius.
     */
    Sphere(const std::array<real, N> &_center, const real _R)
    : Sphere(util::toUvector(_center), _R)
    {}

    /**
     * @brief Constructor.
     *
     * @param _center Sphere center (along diagonal).
     * @param _R Sphere radius.
     */
    Sphere(const real _center, const real _R)
    : Sphere(uvector<real,N>(_center), _R)
    {}

    /**
     * @brief Constructor for sphere centered at (0.5, 0.5, ...)
     *
     * @param _center Sphere center (along diagonal).
     * @param _R Sphere radius.
     */
    Sphere(const real _R)
    : Sphere(0.5, _R)
    {}

    /// Sphere's center.
    const uvector<real, N> center;      
    /// Sphere's radius.
    const real R;

    /**
     * @brief Evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function is evaluated.
     * @return Function value at @p _x.
     */
    template<typename T>
    T operator()(const uvector<T, N>& _x) const
    {
        T val{-R * R};
        for(int dir = 0; dir < N; ++dir)
            val += util::sqr(_x(dir) - center(dir));
        return val;
    }

    /**
     * @brief Gradient evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function's gradient is evaluated.
     * @return Function gradient at @p _x.
     */
    template<typename T>
    uvector<T, N> grad(const uvector<T, N>& _x) const
    {
        uvector<T, N> g;
        for(int dir = 0; dir < N; ++dir)
            g(dir) = 2.0 * (_x(dir) - center(dir));
        return g;
    }

    /**
     * @brief Hessian evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function's hessian is evaluated.
     * @return Function hessian at @p _x.
     */
    template<typename T>
    uvector<T, (N*(N+1))/2> hessian(const uvector<T, N>& _x) const
    {
        uvector<T, (N*(N+1))/2> hess;
        for(int i = 0, ij = 0; i < N; ++i)
        {
            for(int j = i; j < N; ++j, ++ij)
                hess(ij) = (i == j) ? 2.0 : 0.0;
        }
        return hess;
    }
};

/**
 * @brief Infinite 3D cylinder along the z-axis.
 * 
 * The function is defined by its radius.
 * The function presents a negative sign around the z-axis,
 * and positive far away. At a radius distance from the
 * z-axis, the function vanishes.
 * 
 */
struct Cylinder
{
    /**
     * @brief Constructor.
     * 
     * @param _R Cylinder's radius.
     */
    Cylinder(const algoim::real _R) : R(_R)
    {}

    /// Cylinder's radius.
    const algoim::real R;

    /**
     * @brief Evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function is evaluated.
     * @return Function value at @p _x.
     */
    template<typename T>
    T operator() (const algoim::uvector<T,3>& _x) const
    {
        return _x(0)*_x(0) + _x(1)*_x(1) - R * R;
    }

    /**
     * @brief Gradient evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function's gradient is evaluated.
     * @return Function gradient at @p _x.
     */
    template<typename T>
    algoim::uvector<T,3> grad(const algoim::uvector<T,3>& _x) const
    {
        return algoim::uvector<T,3>(2.0*_x(0), 2.0*_x(1), 0.0);
    }

    /**
     * @brief Hessian evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function's hessian is evaluated.
     * @return Function hessian at @p _x.
     */
    template<typename T>
    uvector<T, 6> hessian(const uvector<T, 3>& _x) const
    {
        return uvector<T, 6>(2.0, 0.0, 0.0, 2.0, 0.0, 0.0);
    }
};

/**
 * @brief Dimension independent ellipsoidal function.
 * The function is defined by the ellipsoid's semi-axes and centered at the origin.
 * The function presents a negative sign around the origin,
 * and positive far away.
 * 
 * @tparam N_ Parametric dimension.
 */
template<int N>
struct Ellipsoid
{
    /**
     * @brief Constructor.
     * 
     * @param _semi_axes Semi-axes length along the Cartesian axes.
     */
    Ellipsoid(const uvector<real, N> &_semi_axes)
    : semi_axes(_semi_axes) {}

    /**
     * @brief Default constructor.
     * 
     * Semi-axes are set to (1, 1/2, 1/3, ...)
     */
    Ellipsoid()
    : semi_axes()
    {
        for(int i = 0; i < N; ++i)
            semi_axes(i) = 1.0 / real(i + 1);
    }

    /// Ellipsoid's semi-axes.
    uvector<real, N> semi_axes;

    /**
     * @brief Evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function is evaluated.
     * @return Function value at @p _x.
     */
    template<typename T>
    T operator() (const algoim::uvector<T,N>& _x) const
    {
        return sqrnorm(_x / semi_axes) - 1.0;
    }

    /**
     * @brief Gradient evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function's gradient is evaluated.
     * @return Function gradient at @p _x.
     */
    template<typename T>
    algoim::uvector<T,N> grad(const algoim::uvector<T,N>& _x) const
    {
        algoim::uvector<T, N> g;
        for(int dir = 0; dir < N; ++dir)
            g(dir) = 2.0 * _x(dir) / util::sqr(semi_axes(dir));
        return g;
    }

    /**
     * @brief Hessian evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function's hessian is evaluated.
     * @return Function hessian at @p _x.
     */
    template<typename T>
    uvector<T, (N*(N+1))/2> hessian(const uvector<T, N>& _x) const
    {
        uvector<T, (N*(N+1))/2> hess;
        for(int i = 0, ij = 0; i < N; ++i)
        {
            for(int j = i; j < N; ++j, ++ij)
                hess(ij) = (i == j) ? 2.0 / util::sqr(semi_axes(i)) : 0.0;
        }
        return hess;
    }
};

/**
 * @brief Dimension independent square function.
 * The function is defined by a center and a radius.
 * The function presents a negative sign around the center,
 * and positive far away. At a radius distance from the
 * center, the function vanishes.
 * 
 * @tparam N_ Parametric dimension.
 * 
 * @warning So far, only implemented for the 2D case.
 */
template <int N_>
struct Square
{
    /// Function's parametric dimension.
    static const int N = N_;

    /**
     * @brief Returns the sign of the given value.
     * 
     * @tparam T Type of the value.
     * @param _val Value to be tested.
     * @return +1 is @p _val is positive, -1 if it is negative, 0 otherwise.
     */
    template<typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
    int sgn(const T &_val) const
    {
      return (T(0.0) < _val) - (_val < T(0.0));
    }

    /**
     * @brief Returns the sign of the given value.
     * 
     * @tparam T Type of the value.
     * @param _val Value to be tested.
     * @return +1 is @p _val is positive, -1 if it is negative, 0 otherwise.
     */
    template<typename T, std::enable_if_t<!std::is_arithmetic_v<T>, bool> = true>
    int sgn(const T &val) const
    {
      return val.sign();
    }

    /**
     * @brief Evaluator operator.
     *
     * @param _x Point at which the function is evaluated.
     * @return Function value at @p _x.
     * 
     * @note 2D real case implementation.
     */
    real operator() (const uvector<real,2> &_x) const
    {
      static_assert(N == 2, "Square function only implemented in 2D.");
      const real v = std::max(std::abs(_x(0)) - 1, std::abs(_x(1)) - 1);
      return v;
    }

    /**
     * @brief Evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function is evaluated.
     * @return Function value at @p _x.
     */
    template<typename T>
    T operator() (const uvector<T,2> &_x) const
    {
      const T dx = _x(0), dy = _x(1);
      const real a = std::abs(dx.alpha) - 1;
      const real b = std::abs(dy.alpha) - 1;
      if (a > b)
        return dx*sgn(dx) - 1;
      else
        return dy*sgn(dy) - 1;
    }

  /**
   * @brief Gradient evaluator operator.
   *
   * @tparam T Input and output's type.
   * @param _x Point at which the function's gradient is evaluated.
   * @return Function gradient at @p _x.
   */
  uvector<real,2> grad(const uvector<real,2>& x) const
  {
    const real dx = x(0), dy = x(1);
    const real a = std::abs(dx) - 1;
    const real b = std::abs(dy) - 1;
    if (a > b)
      return uvector<real,2>(sgn(dx), 0.0);
    else
      return uvector<real,2>(0.0, sgn(dy));
  }

  /**
   * @brief Gradient evaluator operator.
   *
   * @tparam T Input and output's type.
   * @param _x Point at which the function's gradient is evaluated.
   * @return Function gradient at @p _x.
   */
  template<typename T>
  uvector<T,2> grad(const uvector<T,2>& x) const
  {
    const T dx = x(0), dy = x(1);
    const real a = std::abs(dx.alpha) - 1;
    const real b = std::abs(dy.alpha) - 1;
    if (a > b)
      return uvector<T,2>(sgn(dx), T(0));
    else
      return uvector<T,2>(T(0), sgn(dy));
  }
};
/**
 * @brief Dimension independent thickness function.
 * 
 * @tparam N_ Parametric dimension.
 */
template<int N_>
struct Thickness
{
    /// Function's parametric dimension.
    static const int N = N_;

    /**
     * @brief Default constructor. Sets constant value to 0.5.
     */
    Thickness() : Thickness(0.5)
    {}

    /**
     * @brief Constructor.
     * 
     * @param _value Constant value.
     */
    Thickness(const real &_value) : value(_value)
    {}

    /// Thickness' constant value.
    const real value;

    /**
     * @brief Evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function is evaluated.
     * @return Function value at @p _x (the constant value).
     */
    template<typename T>
    T operator() (const uvector<T, N>& _x) const
    {
        return T(value) * T(value);
    }

    /**
     * @brief Hessian evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function's hessian is evaluated.
     * @return Function hessian at @p _x (zero, as the function is constant).
     */
    template<typename T>
    uvector<T, (N*(N+1))/2> hessian(const uvector<T, N>& _x) const
    {
        return uvector<T, (N*(N+1))/2>(0.0);
    }
};

/**
 * @brief Dimension independent constant function.
 * 
 * @tparam N_ Parametric dimension.
 */
template<int N_>
struct Constant
{
    /// Function's parametric dimension.
    static const int N = N_;

    /**
     * @brief Default constructor. Sets constant value to 0.5.
     */
    Constant() : Constant(0.5)
    {}

    /**
     * @brief Constructor.
     * 
     * @param _value Constant value.
     */
    Constant(const real &_value) : value(_value)
    {}

    const real value;

    /**
     * @brief Evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function is evaluated.
     * @return Function value at @p _x (the constant value).
     */
    template<typename T>
    T operator() (const uvector<T, N>& _x) const
    {
        return T(value);
    }

    /**
     * @brief Gradient evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function's gradient is evaluated.
     * @return Function gradient at @p _x (zero, as the function is constant).
     */
    template<typename T>
    uvector<T, N> grad(const uvector<T, N>& _x) const
    {
        return uvector<T, N>(0.0);
    }

    /**
     * @brief Hessian evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function's hessian is evaluated.
     * @return Function hessian at @p _x (zero, as the function is constant).
     */
    template<typename T>
    uvector<T, (N*(N+1))/2> hessian(const uvector<T, N>& _x) const
    {
        return uvector<T, (N*(N+1))/2>(0.0);
    }
};

// /**
//  * @brief 1-linear function (by convention, among dim = 0).
//  * 
//  * @tparam N_ Parametric dimension.
//  * @warning To be removed, as it is a subcase of dLinear.
//  */
// template<int N_>
// struct Linear
// {
//     /// Function's parametric dimension.
//     static const int N = N_;

//     /**
//      * @brief Constructor.
//      * 
//      * @param _coefs Constant value.
//      */
//     Linear(const std::vector<real> &_coefs) : a0(_coefs[0]), a1(_coefs[1])
//     {}

//     const real a0;
//     const real a1;

//     /**
//      * @brief Evaluator operator.
//      *
//      * @tparam T Input and output's type.
//      * @param _x Point at which the function is evaluated.
//      * @return Function value at @p _x (the constant value).
//      */
//     template<typename T>
//     T operator() (const uvector<T, N>& _x) const
//     {
//         return a0 + (a1 - a0) * _x(0);
//     }

//     /**
//      * @brief Gradient evaluator operator.
//      *
//      * @tparam T Input and output's type.
//      * @param _x Point at which the function's gradient is evaluated.
//      * @return Function gradient at @p _x.
//      */
//     template<typename T>
//     uvector<T, N> grad(const uvector<T, N>& _x) const
//     {
//         uvector<T, N> grad;
//         grad(0) = a1 - a0;
//         return grad;
//     }

//     /**
//      * @brief Hessian evaluator operator.
//      *
//      * @tparam T Input and output's type.
//      * @param _x Point at which the function's hessian is evaluated.
//      * @return Function gradient at @p _x (zero, as the function is linear).
//      */
//     template<typename T>
//     uvector<T, N> hessian(const uvector<T, N>& _x) const
//     {
//         uvector<T, N> grad;
//         grad(0) = a1 - a0;
//         return grad;
//     }
// };

/**
 * @brief N-linear function.
 * 
 * @tparam N_ Parametric dimension.
 */
template<int N_>
struct dLinear
{
    /// Function's parametric dimension.
    static const int N = N_;

    /// Number of coefficients.
    static const int NC = detail::pow(2, N_);

    /**
     * @brief Constructor.
     * 
     * @param _coefs Constant value.
     */
    dLinear(const uvector<real,NC> &_coefs)
    : coefs(_coefs)
    {}

    /**
     * @brief Constructor.
     * 
     * @param _coefs Constant value.
     */
    dLinear(const std::array<real,NC> &_coefs)
    : dLinear(util::toUvector(_coefs))
    {}

    const uvector<real, NC> coefs;


    /**
     * @brief Evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function is evaluated.
     * @return Function value at @p _x (the constant value).
     */
    template<typename T>
    T operator() (const uvector<T, N>& _x) const
    {
        if constexpr (N == 2) {
            return coefs[0] * (1-_x(0)) * (1-_x(1)) + coefs[1]*_x(0)*(1-_x(1)) + coefs[2] * _x(1) * (1-_x(0)) + coefs[3]*_x(0)*_x(1);       
        }
        else {
            return coefs[0] * (1-_x(0)) * (1-_x(1)) * (1-_x(2)) + coefs[1] * _x(0) * (1-_x(1)) * (1-_x(2)) + coefs[2] * _x(1) * (1-_x(0)) * (1-_x(2)) + coefs[3] * _x(0) * _x(1) * (1-_x(2)) \
                 + coefs[4] * (1-_x(0)) * (1-_x(1)) * _x(2)     + coefs[5] * _x(0) * (1-_x(1)) * _x(2)     + coefs[6] * _x(1) * (1-_x(0)) * _x(2)     + coefs[7] * _x(0) * _x(1) * _x(2);
        }
    }

    /**
     * @brief Gradient evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function's gradient is evaluated.
     * @return Function gradient at @p _x.
     */
    template<typename T>
    uvector<T, N> grad(const uvector<T, N>& _x) const
    {
        uvector<T, N> grad;
        if constexpr (N == 2) {
            grad(0) = (coefs[3]-coefs[2]) * _x(1) + (coefs[1]-coefs[0]) * (1-_x(1));
            grad(1) = (coefs[3]-coefs[1]) * _x(0) + (coefs[2]-coefs[0]) * (1-_x(0));
        }
        else {
            grad(0) = (coefs[3]-coefs[2]) * _x(1) * (1-_x(2)) + (coefs[1]-coefs[0]) * (1-_x(1)) * (1-_x(2)) + (coefs[7]-coefs[6]) * _x(1) * _x(2) + (coefs[5]-coefs[4]) * (1-_x(1)) * _x(2);
            grad(1) = (coefs[3]-coefs[1]) * _x(0) * (1-_x(2)) + (coefs[2]-coefs[0]) * (1-_x(0)) * (1-_x(2)) + (coefs[7]-coefs[5]) * _x(0) * _x(2) + (coefs[6]-coefs[4]) * (1-_x(0)) * _x(2);
            grad(2) = (coefs[5]-coefs[1]) * _x(0) * (1-_x(1)) + (coefs[4]-coefs[0]) * (1-_x(0)) * (1-_x(1)) + (coefs[7]-coefs[3]) * _x(0) * _x(1) + (coefs[6]-coefs[2]) * (1-_x(0)) * _x(1);
        }
        
        return grad;
    }

    /**
     * @brief Hessian evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function's hessian is evaluated.
     * @return Function hessian at @p _x.
     */
    template<typename T>
    uvector<T, (N*(N+1))/2> hessian(const uvector<T, N>& _x) const
    {
        uvector<T, (N*(N+1))/2> hess;
        if constexpr (N == 2) {
            hess(0) = 0.0;
            hess(1) = coefs[0] - coefs[1] - coefs[2] + coefs[3];
            hess(2) = 0.0;
        }
        else {
            hess(0) = 0.0;
            hess(1) = (coefs[3]-coefs[2]) * (1-_x(2)) - (coefs[1]-coefs[0]) * (1-_x(2)) + (coefs[7]-coefs[6]) * _x(2) - (coefs[5]-coefs[4]) * _x(2);
            hess(2) = -(coefs[3]-coefs[2]) * _x(1) - (coefs[1]-coefs[0]) * (1-_x(1)) + (coefs[7]-coefs[6]) * _x(1) + (coefs[5]-coefs[4]) * (1-_x(1));
            hess(3) = 0.0;
            hess(4) = -(coefs[3]-coefs[1]) * _x(0) - (coefs[2]-coefs[0]) * (1-_x(0)) + (coefs[7]-coefs[5]) * _x(0) + (coefs[6]-coefs[4]) * (1-_x(0));
            hess(5) = 0.0;
        }
        
        return hess;
    }
};

/**
 * @brief This function represent as gyroid of type @t G
 * with a given porosity of function type @t P.
 * 
 * @tparam G Gyroid type.
 * @tparam P Porosity type.
 * @tparam N_ Parametric dimension.
 * 
 * @note This function (only) works both in 2D and 3D, even
 * if the gyroid function @t G is only defined in 3D.
 */
template<typename G, int N_, typename P = Constant<N_>>
struct Gyroid
{
    /// Function's parametric dimension.
    static const int N = N_;

    /**
     * @brief Default constructor.
     */
    Gyroid() : Gyroid(P()) {}

    /**
     * @brief Constructor.
     * @p _porosity Porosity function.
     * @note Gyroid periods are set to 1.
     */
    Gyroid(const P &_porosity) : Gyroid(_porosity, 1.0) {}

    /**
     * @brief Constructor.
     * @p _porosity Porosity function.
     * @p _m Gyroid periods.
     */
    Gyroid(const P &_porosity,
           const uvector<real, 3> &_m)
    : porosity(_porosity), m(_m)
    {
        static_assert(N == 2 || N == 3, "Not implemented.");
    }

    /**
     * @brief Constructor.
     * @p _porosity Porosity function.
     * @p _m Gyroid periods.
     */
    Gyroid(const P &_porosity,
           const std::array<real, N> &_m)
    : Gyroid(_porosity, util::toUvector(_m))
    {}

    /// Porosity function.
    const P porosity;
    // Gyroid periods.
    const uvector<real, 3> m;

    /**
     * @brief Evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function is evaluated.
     * @return Function value at @p _x.
     */
    template<typename T>
    T operator() (const uvector<T, N> &_x) const
    {
        return G::eval(m, detail::to_3D(_x)) - porosity(_x);
    }

    /**
     * @brief Gradient evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function's gradient is evaluated.
     * @return Function gradient at @p _x.
     */
    template<typename T>
    uvector<T, N> grad(const uvector<T, N> &_x) const
    {
        const auto g3D = G::eval_grad(m, detail::to_3D(_x));

        uvector<T, N> g;
        if constexpr (N == 2)
            g = remove_component(g3D, 2);
        else // if constexpr (N == 3)
            g = g3D;

        return g - porosity.grad(_x);
    }

    /**
     * @brief Gradient evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function's gradient is evaluated.
     * @return Function gradient at @p _x.
     */
    template<typename T>
    uvector<T, (N==3)?6:3> hessian(const uvector<T, N> &_x) const
    {
        static_assert(N == 2 || N == 3, "Not implemented.");

        const auto h3D = G::eval_hessian(m, detail::to_3D(_x));

        uvector<T, (N==3)?6:3> h;
        if constexpr (N == 2)
        {
            h(0) = h3D(0);
            h(1) = h3D(3);
            h(2) = h3D(1);
        }
        else // if constexpr (N == 3)
            h = h3D;

        return h - porosity.hessian(_x);
    }
};

/**
 * @brief Schoen's gyroid function in 3D.
 * Defined as
 *   f(x,y,z,m,n,q) = sin(2 pi m x) * cos(2 pi n y) + sin(2 pi n y) * cos(2 pi q z) + sin(2 pi q z) * cos(2 pi m x)
 * this is a triply periodic function with period (m, n, q).
 * 
 * See https://en.wikipedia.org/wiki/Gyroid
 */
struct Schoen
{
    /// Function's parametric dimension.
    static const int N = 3;

    /**
     * @brief Evaluates the function at point @p _x for the periods @p _m.
     * 
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function is evaluated.
     * @return Value of the function at @p _x.
     */
    template<typename T>
    static T eval(const uvector<real, 3> &_m,
                  const uvector<T, 3> &_x)
    {
        const uvector<T, 3> pi_2_m_x = (2.0 * util::pi) * _m * _x;
        return sin(pi_2_m_x(0)) * cos(pi_2_m_x(1)) + sin(pi_2_m_x(1)) * cos(pi_2_m_x(2)) + sin(pi_2_m_x(2)) * cos(pi_2_m_x(0));
    }

    /**
     * @brief Computes the function's gradient @p _x for periods @p _m.
     *
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function's gradient is evaluated.
     * @return Function gradient at @p _x.
     */
    template<typename T>
    static uvector<T, 3> eval_grad(const uvector<real, 3> &_m,
                                           const uvector<T, 3> &_x)
    {
        const uvector<real, 3> pi_2_m = 2.0 * util::pi * _m;
        const uvector<T, 3> pi_2_m_x = pi_2_m * _x;

        uvector<T, 3> g;
        g(0) =  cos(pi_2_m_x(0)) * cos(pi_2_m_x(1)) - sin(pi_2_m_x(0)) * sin(pi_2_m_x(2));
        g(1) =  cos(pi_2_m_x(1)) * cos(pi_2_m_x(2)) - sin(pi_2_m_x(1)) * sin(pi_2_m_x(0));
        g(2) =  cos(pi_2_m_x(2)) * cos(pi_2_m_x(0)) - sin(pi_2_m_x(2)) * sin(pi_2_m_x(1));

        return pi_2_m * g;
    }

    /**
     * @brief Computes the function's hessian @p _x for periods @p _m.
     *
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function's hessian is evaluated.
     * @return Function hessian at @p _x.
     */
    template<typename T>
    static uvector<T, 6> eval_hessian(const uvector<real, 3> &_m,
                                      const uvector<T, 3> &_x)
    {
        const uvector<real, 3> pi_2_m = 2.0 * util::pi * _m;

        algoim::uvector<T,6> hess;

        hess(0) = pi_2_m(0) * pi_2_m(0) * (-sin(pi_2_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) - sin(pi_2_m(2) * _x(2)) * cos(pi_2_m(0) * _x(0)));
        hess(1) = pi_2_m(0) * pi_2_m(1) * (-cos(pi_2_m(0) * _x(0)) * sin(pi_2_m(1) * _x(1)));
        hess(2) = pi_2_m(0) * pi_2_m(2) * (-cos(pi_2_m(2) * _x(2)) * sin(pi_2_m(2) * _x(0)));
        hess(3) = pi_2_m(1) * pi_2_m(1) * (-sin(pi_2_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)) - sin(pi_2_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)));
        hess(4) = pi_2_m(1) * pi_2_m(2) * (-cos(pi_2_m(1) * _x(1)) * sin(pi_2_m(2) * _x(2)));
        hess(5) = pi_2_m(2) * pi_2_m(2) * (-sin(pi_2_m(2) * _x(2)) * cos(pi_2_m(0) * _x(0)) - sin(pi_2_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)));

        return hess;
    }
};

/**
 * @brief Squared version of the Schoen's gyroid function in 3D.
 * Defined as
 *   f(x,y,z,m,n,q) = (sin(2 pi m x) * cos(2 pi n y) + sin(2 pi n y) * cos(2 pi q z) + sin(2 pi q z) * cos(2 pi m x)) ^ 2
 * this is a triply periodic function with period (m, n, q).
 * 
 * See @ref Schoen
 */
struct SchoenSquared
{
    /// Function's parametric dimension.
    static const int N = 3;

    /**
     * @brief Evaluates the function at point @p _x for the periods @p _m.
     * 
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function is evaluated.
     * @return Value of the function at @p _x.
     */
    template<typename T>
    static T eval(const uvector<real, 3> &_m,
                  const uvector<T, 3> &_x)
    {
        return util::sqr(Schoen::eval(_m, _x));
    }


    /**
     * @brief Computes the function's gradient @p _x for periods @p _m.
     *
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function's gradient is evaluated.
     * @return Function gradient at @p _x.
     */
    template<typename T>
    static uvector<T, 3> eval_grad(const uvector<real, 3> &_m,
                                   const uvector<T, 3> &_x)
    {
        return 2.0 * Schoen::eval(_m, _x) * Schoen::eval_grad(_m, _x);
    }

    /**
     * @brief Computes the function's hessian @p _x for periods @p _m.
     *
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function's hessian is evaluated.
     * @return Function hessian at @p _x.
     */
    template<typename T>
    static uvector<T, 6> eval_hessian(const uvector<real, 3> &_m,
                                      const uvector<T, 3> &_x)
    {
        uvector<T,6> hess = 2.0 * Schoen::eval(_m, _x) * Schoen::eval_hessian(_m, _x);

        const uvector<T,3> grad = eval_grad(_m, _x);
        hess(0) += 2.0 * grad(0) * grad(0);
        hess(1) += 2.0 * grad(0) * grad(1);
        hess(2) += 2.0 * grad(0) * grad(2);
        hess(3) += 2.0 * grad(1) * grad(1);
        hess(4) += 2.0 * grad(1) * grad(2);
        hess(5) += 2.0 * grad(2) * grad(2);

        return hess;
    }
};

/**
 * @brief Schoen IWP's function in 3D.
 * Defined as
 *   f(x,y,z,m,n,q) = 2 * (cos(2 pi m x) * cos(2 pi n y) + cos(2 pi n y) * cos(2 pi q z) + cos(2 pi q z) * cos(2 pi m x)) - cos(4 pi m x) - cos(4 pi m n y) - cos(4 pi q z)
 * this is a triply periodic function with period (m, n, q).
 */
struct SchoenIWP
{
    /// Function's parametric dimension.
    static const int N = 3;

    /**
     * @brief Evaluates the function at point @p _x for the periods @p _m.
     * 
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function is evaluated.
     * @return Value of the function at @p _x.
     */
    template<typename T>
    static T eval(const uvector<real, 3> &_m,
                  const uvector<T, 3> &_x)
    {
        const uvector<T, 3> pi_2_m_x = 2.0 * util::pi * _m * _x;
        const uvector<T, 3> pi_4_m_x = 2.0 * pi_2_m_x;

        return 2.0 * (cos(pi_2_m_x(0)) * cos(pi_2_m_x(1))
                    + cos(pi_2_m_x(1)) * cos(pi_2_m_x(2))
                    + cos(pi_2_m_x(2)) * cos(pi_2_m_x(0)))
             - cos(pi_4_m_x(0)) - cos(pi_4_m_x(1)) - cos(pi_4_m_x(2));
    }

    /**
     * @brief Computes the function's gradient @p _x for periods @p _m.
     *
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function's gradient is evaluated.
     * @return Function gradient at @p _x.
     */
    template<typename T>
    static uvector<T, 3> eval_grad(const uvector<real, 3> &_m,
                                   const uvector<T, 3> &_x)
    {
        const uvector<real, 3> pi_2_m = 2.0 * util::pi * _m;
        const uvector<T, 3> pi_2_m_x = pi_2_m * _x;
        const uvector<T, 3> pi_4_m_x = 2.0 * pi_2_m_x;

        uvector<T, 3> g;

        g(0) = sin(pi_4_m_x(0)) - sin(pi_2_m_x(0)) * (cos(pi_2_m_x(1)) + cos(pi_2_m_x(2)));
        g(1) = sin(pi_4_m_x(1)) - sin(pi_2_m_x(1)) * (cos(pi_2_m_x(0)) + cos(pi_2_m_x(2)));
        g(2) = sin(pi_4_m_x(2)) - sin(pi_2_m_x(2)) * (cos(pi_2_m_x(0)) + cos(pi_2_m_x(1)));

        return 2.0 * pi_2_m * g;
    }

    /**
     * @brief Computes the function's hessian @p _x for periods @p _m.
     *
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function's hessian is evaluated.
     * @return Function hessian at @p _x.
     */
    template<typename T>
    static uvector<T, 6> eval_hessian(const uvector<real, 3> &_m,
                                      const uvector<T, 3> &_x)
    {
        const uvector<real, 3> pi_2_m = 2.0 * util::pi * _m;
        const uvector<real, 3> pi_4_m = 4.0 * util::pi * _m;

        algoim::uvector<T,6> hess;
        hess(0) = -pi_2_m(0) * pi_4_m(0) * (cos(pi_2_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) + cos(pi_2_m(2) * _x(2)) * cos(pi_2_m(0) * _x(0)) - 2.0 * cos(pi_4_m(0) * _x(0)));
        hess(1) =  pi_2_m(0) * pi_4_m(1) * sin(pi_2_m(0) * _x(0)) * sin(pi_2_m(1) * _x(1));
        hess(2) =  pi_2_m(0) * pi_4_m(2) * sin(pi_2_m(2) * _x(2)) * sin(pi_2_m(0) * _x(0));
        hess(3) = -pi_2_m(1) * pi_4_m(1) * (cos(pi_2_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) + cos(pi_2_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)) - 2.0 * cos(pi_4_m(1) * _x(1)));
        hess(4) =  pi_2_m(1) * pi_4_m(2) * sin(pi_2_m(1) * _x(1)) * sin(pi_2_m(2) * _x(2));
        hess(5) = -pi_2_m(2) * pi_4_m(2) * (cos(pi_2_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)) + cos(pi_2_m(2) * _x(2)) * cos(pi_2_m(0) * _x(0)) - 2.0 * cos(pi_4_m(2) * _x(2)));

        return hess;
    }
};

/**
 * @brief Schoen FRD's function in 3D.
 * Defined as
 *   f(x,y,z,m,n,q) = 4 * cos(2 pi m x) * cos(2 pi n y) * cos(2 pi q z) - cos(4 pi m x) * cos(4 pi n y) - cos(4 pi n y) * cos(4 pi q z) - cos(4 pi m q z* cos(4 pi m x)
 * this is a triply periodic function with period (m, n, q).
 */
struct SchoenFRD
{
    /// Function's parametric dimension.
    static const int N = 3;

    /**
     * @brief Evaluates the function at point @p _x for the periods @p _m.
     * 
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function is evaluated.
     * @return Value of the function at @p _x.
     */
    template<typename T>
    static T eval(const uvector<real, 3> &_m,
                  const uvector<T, 3> &_x)
    {
        const uvector<real, 3> pi_2_m = 2.0 * util::pi * _m;
        const uvector<T, 3> pi_2_m_x = pi_2_m * _x;
        const uvector<T, 3> pi_4_m_x = 2.0 * pi_2_m_x;

        return 4 * cos(pi_2_m_x(0)) * cos(pi_2_m_x(1)) * cos(pi_2_m_x(2))
                 - cos(pi_4_m_x(0)) * cos(pi_4_m_x(1))
                 - cos(pi_4_m_x(1)) * cos(pi_4_m_x(2))
                 - cos(pi_4_m_x(2)) * cos(pi_4_m_x(0));
    }

    /**
     * @brief Computes the function's gradient @p _x for periods @p _m.
     *
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function's gradient is evaluated.
     * @return Function gradient at @p _x.
     */
    template<typename T>
    static uvector<T, 3> eval_grad(const uvector<real, 3> &_m,
                                           const uvector<T, 3> &_x)
    {
        const uvector<real, 3> pi_2_m = 2.0 * util::pi * _m;
        const uvector<T, 3> pi_2_m_x = pi_2_m * _x;
        const uvector<T, 3> pi_4_m_x = 2.0 * pi_2_m_x;

        uvector<T, 3> g;

        g(0) = -2.0 * sin(pi_2_m_x(0)) * cos(pi_2_m_x(1)) * cos(pi_2_m_x(2))
                    + sin(pi_4_m_x(0)) * cos(pi_4_m_x(1))
                    + cos(pi_4_m_x(2)) * sin(pi_4_m_x(0));

        g(1) = -2.0 * cos(pi_2_m_x(0)) * sin(pi_2_m_x(1)) * cos(pi_2_m_x(2))
                    + cos(pi_4_m_x(0)) * sin(pi_4_m_x(1))
                    + sin(pi_4_m_x(1)) * cos(pi_4_m_x(2));

        g(2) = -2.0 * cos(pi_2_m_x(0)) * cos(pi_2_m_x(1)) * sin(pi_2_m_x(2))
                    + cos(pi_4_m_x(1)) * sin(pi_4_m_x(2))
                    + sin(pi_4_m_x(2)) * cos(pi_4_m_x(0));

        return 2.0 * pi_2_m * g;
    }

    /**
     * @brief Computes the function's hessian @p _x for periods @p _m.
     *
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function's hessian is evaluated.
     * @return Function hessian at @p _x.
     */
    template<typename T>
    static uvector<T, 6> eval_hessian(const uvector<real, 3> &_m,
                                      const uvector<T, 3> &_x)
    {
        const uvector<real, 3> pi_2_m = 2.0 * util::pi * _m;
        const uvector<real, 3> pi_4_m = 4.0 * util::pi * _m;

        algoim::uvector<T,6> hess;

        hess(0) = pi_2_m(0) * pi_4_m(0) * (-2.0 * cos(pi_2_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)) + 2.0 * cos(pi_4_m(0) * _x(0)) * cos(pi_4_m(1) * _x(1)) + 2.0 * cos(pi_4_m(2) * _x(2)) * cos(pi_4_m(0) * _x(0)));
        hess(1) = pi_2_m(0) * pi_4_m(1) * (2.0 * sin(pi_2_m(0) * _x(0)) * sin(pi_2_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)) - 2.0 * sin(pi_4_m(0) * _x(0)) * sin(pi_4_m(1) * _x(1)));
        hess(2) = pi_2_m(0) * pi_4_m(2) * (2.0 * sin(pi_2_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) * sin(pi_2_m(2) * _x(2)) - 2.0 * sin(pi_4_m(2) * _x(2)) * sin(pi_4_m(0) * _x(0)));
        hess(3) = pi_2_m(1) * pi_4_m(1) * (-2.0 * cos(pi_2_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)) + 2.0 * cos(pi_4_m(0) * _x(0)) * cos(pi_4_m(1) * _x(1)) + 2.0 * cos(pi_4_m(1) * _x(1)) * cos(pi_4_m(2) * _x(2)));
        hess(4) = pi_2_m(1) * pi_4_m(2) * (2.0 * cos(pi_2_m(0) * _x(0)) * sin(pi_2_m(1) * _x(1)) * sin(pi_2_m(2) * _x(2)) -2.0 * sin(pi_4_m(1) * _x(1)) * sin(pi_4_m(2) * _x(2)));
        hess(5) = pi_2_m(2) * pi_4_m(2) * (-2.0 * cos(pi_2_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)) + 2.0 * cos(pi_4_m(1) * _x(1)) * cos(pi_4_m(2) * _x(2)) + 2.0 * cos(pi_4_m(2) * _x(2)) * cos(pi_4_m(0) * _x(0)));

        return hess;
    }
};

/**
 * @brief Fischer-Koch S' function in 3D.
 * Defined as
 *   f(x,y,z,m,n,q) = cos(4 pi m x) * sin(2 pi n y) * cos(2 pi q z) + cos(2 pi m x) * cos(4 pi n y) * sin(2 pi q z) + sin(2 pi m x) * cos(2 pi n y) * cos(4 pi q z)
 * this is a triply periodic function with period (m, n, q).
 */
struct FischerKochS
{
    /// Function's parametric dimension.
    static const int N = 3;

    /**
     * @brief Evaluates the function at point @p _x for the periods @p _m.
     * 
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function is evaluated.
     * @return Value of the function at @p _x.
     */
    template<typename T>
    static T eval(const uvector<real, 3> &_m,
                  const uvector<T, 3> &_x)
    {
        const uvector<real, 3> pi_2_m = 2.0 * util::pi * _m;
        const uvector<T, 3> pi_2_m_x = pi_2_m * _x;
        const uvector<T, 3> pi_4_m_x = 2.0 * pi_2_m_x;

        return cos(pi_4_m_x(0)) * sin(pi_2_m_x(1)) * cos(pi_2_m_x(2))
             + cos(pi_2_m_x(0)) * cos(pi_4_m_x(1)) * sin(pi_2_m_x(2))
             + sin(pi_2_m_x(0)) * cos(pi_2_m_x(1)) * cos(pi_4_m_x(2));
    }

    /**
     * @brief Computes the function's gradient @p _x for periods @p _m.
     *
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function's gradient is evaluated.
     * @return Function gradient at @p _x.
     */
    template<typename T>
    static uvector<T, 3> eval_grad(const uvector<real, 3> &_m,
                                           const uvector<T, 3> &_x)
    {
        const uvector<real, 3> pi_2_m = 2.0 * util::pi * _m;
        const uvector<T, 3> pi_2_m_x = pi_2_m * _x;
        const uvector<T, 3> pi_4_m_x = 2.0 * pi_2_m_x;

        uvector<T, 3> g;

        g(0) = -2.0 * sin(pi_4_m_x(0)) * sin(pi_2_m_x(1)) * cos(pi_2_m_x(2))
                    - sin(pi_2_m_x(0)) * cos(pi_4_m_x(1)) * sin(pi_2_m_x(2))
                    + cos(pi_2_m_x(0)) * cos(pi_2_m_x(1)) * cos(pi_4_m_x(2));

        g(1) =        cos(pi_4_m_x(0)) * cos(pi_2_m_x(1)) * cos(pi_2_m_x(2))
              - 2.0 * cos(pi_2_m_x(0)) * sin(pi_4_m_x(1)) * sin(pi_2_m_x(2))
                    - sin(pi_2_m_x(0)) * sin(pi_2_m_x(1)) * cos(pi_4_m_x(2));

        g(2) =      - cos(pi_4_m_x(0)) * sin(pi_2_m_x(1)) * sin(pi_2_m_x(2))
                    + cos(pi_2_m_x(0)) * cos(pi_4_m_x(1)) * cos(pi_2_m_x(2))
              - 2.0 * sin(pi_2_m_x(0)) * cos(pi_2_m_x(1)) * sin(pi_4_m_x(2));

        return pi_2_m * g;
    }

    /**
     * @brief Computes the function's hessian @p _x for periods @p _m.
     *
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function's hessian is evaluated.
     * @return Function hessian at @p _x.
     */
    template<typename T>
    static uvector<T, 6> eval_hessian(const uvector<real, 3> &_m,
                                      const uvector<T, 3> &_x)
    {
        const uvector<real,3> pi_2_m = 2.0 * util::pi * _m;
        const uvector<real,3> pi_4_m = 4.0 * util::pi * _m;

        algoim::uvector<T,6> hess;
        hess(0) = -pi_4_m(0) * pi_4_m(0) * cos(pi_4_m(0) * _x(0)) * sin(pi_2_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)) - pi_2_m(0) * pi_2_m(0) * cos(pi_2_m(0) * _x(0)) * cos(pi_4_m(1) * _x(1)) * sin(pi_2_m(2) * _x(2)) - pi_2_m(0) * pi_2_m(0) * sin(pi_2_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) * cos(pi_4_m(2) * _x(2));
        hess(1) = -pi_4_m(0) * pi_2_m(1) * sin(pi_4_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)) + pi_2_m(0) * pi_4_m(1) * sin(pi_2_m(0) * _x(0)) * sin(pi_4_m(1) * _x(1)) * sin(pi_2_m(2) * _x(2)) - pi_2_m(0) * pi_2_m(1) * cos(pi_2_m(0) * _x(0)) * sin(pi_2_m(1) * _x(1)) * cos(pi_4_m(2) * _x(2));
        hess(2) = pi_4_m(0) * pi_2_m(2) * sin(pi_4_m(0) * _x(0)) * sin(pi_2_m(1) * _x(1)) * sin(pi_2_m(2) * _x(2)) - pi_2_m(0) * pi_2_m(2) * sin(pi_2_m(0) * _x(0)) * cos(pi_4_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)) - pi_2_m(0) * pi_4_m(2) * cos(pi_2_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) * sin(pi_4_m(2) * _x(2));
        hess(3) = -pi_2_m(1) * pi_2_m(1) * cos(pi_4_m(0) * _x(0)) * sin(pi_2_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)) - pi_4_m(1) * pi_4_m(1) * cos(pi_2_m(0) * _x(0)) * cos(pi_4_m(1) * _x(1)) * sin(pi_2_m(2) * _x(2)) - pi_2_m(1) * pi_2_m(1) * sin(pi_2_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) * cos(pi_4_m(2) * _x(2));
        hess(4) = -pi_2_m(1) * pi_2_m(2) * cos(pi_4_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) * sin(pi_2_m(2) * _x(2)) - pi_4_m(1) * pi_2_m(2) * cos(pi_2_m(0) * _x(0)) * sin(pi_4_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)) + pi_2_m(1) * pi_4_m(2) * sin(pi_2_m(0) * _x(0)) * sin(pi_2_m(1) * _x(1)) * sin(pi_4_m(2) * _x(2));
        hess(5) = -pi_2_m(2) * pi_2_m(2) * cos(pi_4_m(0) * _x(0)) * sin(pi_2_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)) - pi_2_m(2) * pi_2_m(2) * cos(pi_2_m(0) * _x(0)) * cos(pi_4_m(1) * _x(1)) * sin(pi_2_m(2) * _x(2)) - pi_4_m(2) * pi_4_m(2) * sin(pi_2_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) * cos(pi_4_m(2) * _x(2));

        return hess;
    }
};

/**
 * @brief Schwarz Diamond's function in 3D.
 * Defined as
 *   f(x,y,z,m,n,q) = cos(2 pi m x) * cos(2 pi n y) * cos(2 pi q z) - sin(2 pi m x) * sin(2 pi n y) * sin(2 pi q z)
 * this is a triply periodic function with period (m, n, q).
 */
struct SchwarzDiamond
{
    /// Function's parametric dimension.
    static const int N = 3;

    /**
     * @brief Evaluates the function at point @p _x for the periods @p _m.
     * 
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function is evaluated.
     * @return Value of the function at @p _x.
     */
    template<typename T>
    static T eval(const uvector<real, 3> &_m,
                  const uvector<T, 3> &_x)
    {
        const uvector<real, 3> pi_2_m = 2.0 * util::pi * _m;
        const uvector<T, 3> pi_2_m_x = pi_2_m * _x;

        return cos(pi_2_m_x(0)) * cos(pi_2_m_x(1)) * cos(pi_2_m_x(2))
             - sin(pi_2_m_x(0)) * sin(pi_2_m_x(1)) * sin(pi_2_m_x(2));
    }

    /**
     * @brief Computes the function's gradient @p _x for periods @p _m.
     *
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function's gradient is evaluated.
     * @return Function gradient at @p _x.
     */
    template<typename T>
    static uvector<T, 3> eval_grad(const uvector<real, 3> &_m,
                                           const uvector<T, 3> &_x)
    {
        const uvector<real, 3> pi_2_m = 2.0 * util::pi * _m;
        const uvector<T, 3> pi_2_m_x = pi_2_m * _x;

        uvector<T, 3> g;

        g(0) = sin(pi_2_m_x(0)) * cos(pi_2_m_x(1)) * cos(pi_2_m_x(2))
             + cos(pi_2_m_x(0)) * sin(pi_2_m_x(1)) * sin(pi_2_m_x(2));

        g(1) = cos(pi_2_m_x(0)) * sin(pi_2_m_x(1)) * cos(pi_2_m_x(2))
             + sin(pi_2_m_x(0)) * cos(pi_2_m_x(1)) * sin(pi_2_m_x(2));

        g(2) = cos(pi_2_m_x(0)) * cos(pi_2_m_x(1)) * sin(pi_2_m_x(2))
             + sin(pi_2_m_x(0)) * sin(pi_2_m_x(1)) * cos(pi_2_m_x(2));

        return -pi_2_m * g;
    }

    /**
     * @brief Computes the function's hessian @p _x for periods @p _m.
     *
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function's hessian is evaluated.
     * @return Function hessian at @p _x.
     */
    template<typename T>
    static uvector<T, 6> eval_hessian(const uvector<real, 3> &_m,
                                      const uvector<T, 3> &_x)
    {
        const uvector<real,3> pi_2_m = 2.0 * util::pi * _m;

        algoim::uvector<T,6> hess;
        hess(0) = - pi_2_m(0) * pi_2_m(0) * (cos(pi_2_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)) - sin(pi_2_m(0) * _x(0)) * sin(pi_2_m(1) * _x(1)) * sin(pi_2_m(2) * _x(2)));
        hess(1) = - pi_2_m(0) * pi_2_m(1) * (-sin(pi_2_m(0) * _x(0)) * sin(pi_2_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)) + cos(pi_2_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) * sin(pi_2_m(2) * _x(2)));
        hess(2) = - pi_2_m(0) * pi_2_m(2) * (-sin(pi_2_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) * sin(pi_2_m(2) * _x(2)) + cos(pi_2_m(0) * _x(0)) * sin(pi_2_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)));
        hess(3) = - pi_2_m(1) * pi_2_m(1) * (cos(pi_2_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)) - sin(pi_2_m(0) * _x(0)) * sin(pi_2_m(1) * _x(1)) * sin(pi_2_m(2) * _x(2)));
        hess(4) = - pi_2_m(1) * pi_2_m(2) * (-cos(pi_2_m(0) * _x(0)) * sin(pi_2_m(1) * _x(1)) * sin(pi_2_m(2) * _x(2)) + sin(pi_2_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)));
        hess(5) = - pi_2_m(2) * pi_2_m(2) * (cos(pi_2_m(0) * _x(0)) * cos(pi_2_m(1) * _x(1)) * cos(pi_2_m(2) * _x(2)) - sin(pi_2_m(0) * _x(0)) * sin(pi_2_m(1) * _x(1)) * sin(pi_2_m(2) * _x(2)));

        return hess;
    }
};

/**
 * @brief Schwarz Primitive's function in 3D.
 * Defined as
 *   f(x,y,z,m,n,q) = cos(2 pi m_x(0)) + cos(2 pi n y) + cos(2 pi q z)
 * this is a triply periodic function with period (m, n, q).
 */
struct SchwarzPrimitive
{
    /// Function's parametric dimension.
    static const int N = 3;

    /**
     * @brief Evaluates the function at point @p _x for the periods @p _m.
     * 
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function is evaluated.
     * @return Value of the function at @p _x.
     */
    template<typename T>
    static T eval(const uvector<real, 3> &_m,
                  const uvector<T, 3> &_x)
    {
        const uvector<real, 3> pi_2_m = 2.0 * util::pi * _m;
        const uvector<T, 3> pi_2_m_x = pi_2_m * _x;

        return cos(pi_2_m_x(0)) + cos(pi_2_m_x(1)) + cos(pi_2_m_x(2));
    }

    /**
     * @brief Computes the function's gradient @p _x for periods @p _m.
     *
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function's gradient is evaluated.
     * @return Function gradient at @p _x.
     */
    template<typename T>
    static uvector<T, 3> eval_grad(const uvector<real, 3> &_m,
                                           const uvector<T, 3> &_x)
    {
        const uvector<real, 3> pi_2_m = 2.0 * util::pi * _m;
        const uvector<T, 3> pi_2_m_x = pi_2_m * _x;

        uvector<T, 3> g;

        g(0) = sin(pi_2_m_x(0));
        g(1) = sin(pi_2_m_x(1));
        g(2) = sin(pi_2_m_x(2));

        return -pi_2_m * g;
    }

    /**
     * @brief Computes the function's hessian @p _x for periods @p _m.
     *
     * @tparam T Input and output's type.
     * @param _m Periods along the three parametric direction.
     * @param _x Point at which the function's hessian is evaluated.
     * @return Function hessian at @p _x.
     */
    template<typename T>
    static uvector<T, 6> eval_hessian(const uvector<real, 3> &_m,
                                      const uvector<T, 3> &_x)
    {
        const uvector<real,3> pi_2_m = 2.0 * util::pi * _m;
        const uvector<real,3> pi2_4_m2 = util::sqr(pi_2_m);

        algoim::uvector<T,6> hess;

        hess(0) = - pi2_4_m2(0) * cos(pi_2_m(0) * _x(0));
        hess(1) = 0.0;
        hess(2) = 0.0;
        hess(3) = - pi2_4_m2(1) * cos(pi_2_m(1) * _x(1));
        hess(4) = 0.0;
        hess(5) = - pi2_4_m2(2) * cos(pi_2_m(2) * _x(2));

        return hess;
    }
};
/**
 * @brief This function computes the dual (negative) of a given function.
 * 
 * @tparam F Type of the function whose dual is computed.
 */
template<typename F>
struct DualFunction
{
    /// Function's parametric dimension.
    static const int N = F::N;

    /**
     * @brief Constructor.
     * 
     * @param _f Function whose dual is computed.
     */
    DualFunction(const F &_f) : f(_f) {}

    // Function whose dual is computed.
    const F f;

    /**
     * @brief Evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function is evaluated.
     * @return Function value at @p _x.
     */
    template<typename T>
    T operator() (const uvector<T, N> &_x) const
    {
        return - f(_x);
    }

    /**
     * @brief Gradient evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function's gradient is evaluated.
     * @return Function gradient at @p _x.
     */
    template<typename T>
    uvector<T, N> grad(const uvector<T, N> &_x) const
    {
        return - f.grad(_x);
    }

    /**
     * @brief Hessian evaluator operator.
     *
     * @tparam T Input and output's type.
     * @param _x Point at which the function's hessian is evaluated.
     * @return Function hessian at @p _x.
     */
    template<typename T>
    uvector<T, (N*(N+1))/2> hessian(const uvector<T, N> &_x) const
    {
        return - f.hessian(_x);
    }
};


} // namespace algoim


#endif // ALGOIM_IMPLICIT_FUNCTIONS_LIB_H