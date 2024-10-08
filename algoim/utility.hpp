#ifndef ALGOIM_UTILITY_HPP
#define ALGOIM_UTILITY_HPP

#include "uvector.hpp"
#include "real.hpp"

#include <array>
#include <cassert>

// Minor utility methods used throughout Algoim

namespace algoim::util 
{
    static_assert(std::is_same_v<real, double>, "Warning: pi constant may require redefining when real != double");
    static constexpr double pi = 3.141592653589793238462643383279502884197169399375105820974944592307816;

    // square of u
    template<typename T>
    constexpr auto sqr(T u)
    {
        return u*u;
    }

    // cube of u
    template<typename T>
    constexpr auto cube(T u)
    {
        return u*u*u;
    }

    // sign of u:
    //   -1, if u < 0,
    //   +1, if u > 0,
    //    0, otherwise
    template <typename T>
    constexpr int sign(T u) noexcept
    {
        return (T(0) < u) - (u < T(0));
    }

    // collapse an N-dimensional multi-index into a scalar integer according to its location
    // in an N-dimensional grid of the given extent, such that the lowest dimension iterates
    // the slowest, and the highest dimension corresponds to the inner-most loop, consistent
    // with MultiLoop semantics. For example,
    //    furl( {i,j}, {m,n} ) = i*n + j
    //    furl( {i,j,k}, {m,n,o} ) = i*n*o + j*o + k
    template<int N>
    int furl(const uvector<int,N>& i, const uvector<int,N>& ext)
    {
        int ind = i(0);
        for (int j = 1; j < N; ++j)
            ind = ext(j) * ind + i(j);
        return ind;
    }

    // compute a Givens rotation
    template<typename T>
    void givens_get(const T& a, const T& b, T& c, T& s)
    {
        using std::abs;
        using std::sqrt;
        if (b == 0.0)
        {
            c = 1.0;
            s = 0.0;
        }
        else if (abs(b) > abs(a))
        {
            T tmp = a / b;
            s = T(1) / sqrt(1.0 + tmp*tmp);
            c = tmp * s;
        }
        else
        {
            T tmp = b / a;
            c = T(1) / sqrt(1.0 + tmp*tmp);
            s = tmp * c;
        }
    };

    // apply a Givens rotation
    template<typename T>
    void givens_rotate(T& x, T& y, T c, T s)
    {
        T a = x, b = y;
        x =  c * a + s * b;
        y = -s * a + c * b;
    };

    // Transforms a std::array to a uvector.
    template<typename T, int N>
    uvector<T,N> toUvector(const std::array<T, N> &_arr)
    {
        uvector<T,N> vec;
        for(int dir = 0; dir < N; ++dir)
            vec(dir) = _arr[dir];
        return vec;
    }


    /**
     * @brief Struct for comparing two uvectors as "_a < _b".
     * This was defined for using uvectors as key in std::map.
     * 
     * It compares the first component of both vector: if they are
     * different, returns the result of _a(0) <  _b(0); if equal,
     * compares the second component, and so on.
     * 
     * @tparam T Types of the vectors.
     * @tparam N Dimensions of the vectors.
     * @param _a First vector to compare.
     * @param _b Second vector to compare.
     * @return True if the _a is smaller than _b (according to the
     * definition above), false otherwise.
     */
    template<typename T, int N>
    struct UvectorCompare {
        bool operator()(const uvector<T, N> &_a, const uvector<T, N> &_b) const {
            if constexpr (0 < N )
            {
                for(int dir = 0; dir < N; ++dir)
                {
                    if (_a(dir) == _b(dir))
                        continue;
                    return _a(dir) < _b(dir);
                }
            }
            return false;
        }
    };

    /**
     * @brief Computes the flat index from a tensor index for a given size per dimension
     * using counter-lexicographical ordering (higher dimensions faster).
     * 
     * @tparam N Dimension.
     * @param _n Size per dimension.
     * @param _tid Tensor index.
     * @return Flat index.
     */
    template<int N>
    static int toFlatIndex(const uvector<int, N> &_n, const uvector<int,N> &_tid)
    {
        int id = _tid(0);
        for (int i = 1; i < N; ++i)
            id = _n(i) * id + _tid(i);
        return id;
    }

    /**
     * @brief Computes the tensor index from the flat index for a given
     * size per dimension using counter-lexicographical ordering (higher dimensions faster).
     * 
     * @tparam N Dimension.
     * @param _n Size per dimension.
     * @param _id Flat index.
     * @return Tensor index.
     */
    template<int N>
    static uvector<int,N> toTensorIndex(const uvector<int, N> &_n, int _id)
    {
        assert(0 <= _id && _id < prod(_n));

        int s = prod(_n);

        uvector<int, N> tid;
        for(int i = 0; i < N; ++i)
        {
            s /= _n(i);
            tid(i) = _id / s;
            _id -= tid(i) * s;
        }
        return tid;
    }

    /**
     * @brief Computes the flat index from a tensor index for a given size per dimension
     * using lexicographical ordering (lower dimensions faster).
     * 
     * @tparam N Dimension.
     * @param _n Size per dimension.
     * @param _tid Tensor index.
     * @return Flat index.
     */
    template<int N>
    static int toFlatIndexLex(const uvector<int, N> &_n, const uvector<int,N> &_tid)
    {
        int id = _tid(N-1);
        for(int i = N-2; i >= 0; --i)
            id = _n(i) * id + _tid(i);
        return id;
    }

    /**
     * @brief Computes the tensor index from the flat index for a given
     * size per dimension using lexicographical ordering (lower dimensions faster).
     * 
     * @tparam N Dimension.
     * @param _n Size per dimension.
     * @param _id Flat index.
     * @return Tensor index.
     */
    template<int N>
    static uvector<int,N> toTensorIndexLex(const uvector<int, N> &_n, int _id)
    {
        assert(0 <= _id && _id < prod(_n));

        int s = prod(_n);

        uvector<int, N> tid;
        for(int i = N-1; i >= 0; --i)
        {
            s /= _n(i);
            tid(i) = _id / s;
            _id -= tid(i) * s;
        }
        return tid;
    }

} // namespace algoim::util

#endif
