#ifndef ALGOIM_BAND_MATRIX_H
#define ALGOIM_BAND_MATRIX_H

#include "real.hpp"

#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>

namespace algoim
{

struct SymBandMatrixCholesky;

/**
 * @brief Data structure for storing square symmetric band matrices.
 * 
 * Only the main and upper diagonals are stored.
 */
struct SymBandMatrix
{
    /**
     * @brief Constructor.
     * 
     * @param _n Matrix dimension.
     * @param _bw Half-bandwidth (without considering diagonal term).
     */
    SymBandMatrix(const int _n, const int _bw)
    :
    n(_n),
    bw(_bw),
    data(_n * (_bw + 1), 0.0)
    {
        assert(0 < n);
        assert(0 <= bw);
    }

    /**
     * @brief Access the element ( @p _i, @p _j ) of the matrix.
     * 
     * @param _i Row.
     * @param _j Column.
     * @return Constant reference to the value.
     *
     * @warning If the value is not in the band of the matrix,
     * an assert is thrown in debug mode.
     */
    const real &operator()(const int _i, const int _j) const
    {
        const int ii = std::min(_i, _j);
        const int jj = std::abs(_i - _j);
        if (jj > bw)
        {
            assert(jj <= bw);
        }

        assert(0 <= ii && ii < n);

        return data[ii + n * jj];
    }

    /**
     * @brief Access the element ( @p _i, @p _j ) of the matrix.
     * 
     * @param _i Row.
     * @param _j Column.
     * @return Non-constant reference to the value.
     *
     * @warning If the value is not in the band of the matrix,
     * an assert is thrown in debug mode.
     */
    real &operator()(const int _i, const int _j)
    {
        return const_cast<real &>(const_cast<const SymBandMatrix *>(this)->operator()(_i, _j));
    }

    /**
     * @brief Writes the given @p _matrix in the ostream @p _os.
     * 
     * @param _os Stream in which the matrix is written.
     * @param _matrix Matrix to be written.
     * @return Stream in which the matrix is written.
     */
    friend std::ostream& operator<<(std::ostream& _os, const SymBandMatrix& _matrix);

    /// Matrix dimension.
    const int n;
    /// Half-bandwidth (without considering diagonal term).
    const int bw;
    /// Data of the matrix. The diagonal is stored first, the second diagonal with trailing zeros, and so on.
    std::vector<real> data;

};

/**
 * @brief Cholesky factorization of a symmetric banded matrix.
 */
struct SymBandMatrixCholesky : public SymBandMatrix
{
    /**
     * @brief Constructor.
     * @param _matrix Matrix to factorize (assumed to the positive definite)p
     */
    SymBandMatrixCholesky(const SymBandMatrix &_matrix)
    :
    SymBandMatrix(_matrix)
    {
        this->factor();
    }

    /**
     * @brief Solves a linear system A x = b, where A is the matrix
     * whose factorization is stored, and the right-hand side vector b
     * is passed in @p _in_out.
     * 
     * @param _in_out Right-hand side vector b and vector for storing the result x.
     * I.e., the initial information will be overwritten.
     */
    void solve(std::vector<real> &_in_out) const
    {
        assert(static_cast<int>(_in_out.size()) == (this->n));

        real * const ptr_x = _in_out.data();

        forwardSubs(ptr_x, 1);
        backwardSubs(ptr_x, 1);
    }

    /**
     * @brief Solves a linear system A x = b, where A is the matrix
     * whose factorization is stored, and the right-hand side vector @p _b.
     * 
     * @param _b Right-hand side vector.
     * @param _x Solution vector.
     */
    void solve(const std::vector<real> &_b, std::vector<real> &_x) const
    {
        assert(static_cast<int>(_b.size()) == (this->n));
        _x = _b;
        solve(_x);
    }

    /**
     * @brief Solves a linear system A2 x A1 c = b, where x reprensents
     * the Kronecker product between matrices, A2 and A1 are matrices
     * whose factorizations are stored in @p _A2 and @p _A1, respectively,
     * and the right-hand side vector b is stored ni @p _in_out.
     * 
     * @param _A2 Factorization of the matrix A2.
     * @param _A1 Factorization of the matrix A1.
     * @param _in_out Right-hand side vector b and vector for storing the result x.
     * I.e., the initial information will be overwritten.
     */
    static void solveKronecker(const SymBandMatrixCholesky &_A2,
                               const SymBandMatrixCholesky &_A1,
                               std::vector<real> &_in_out)
    {
        const auto n1 = _A1.n;
        const auto n2 = _A2.n;

        assert(static_cast<int>(_in_out.size()) == (n1 * n2));

        real * const ptr_x = _in_out.data();

        for(int i = 0; i < n2; ++i)
        {
            _A1.forwardSubs(ptr_x + i * n1, 1);
            _A1.backwardSubs(ptr_x + i * n1, 1);
        }

        for(int i = 0; i < n1; ++i)
        {
            _A2.forwardSubs(ptr_x + i, n1);
            _A2.backwardSubs(ptr_x + i, n1);
        }
    }

    /**
     * @brief Solves a linear system A3 x A2 x A1 c = b, where x reprensents
     * the Kronecker product between matrices, A3, A2, and A1 are matrices
     * whose factorizations are stored in @p _A3, @p _A2, and @p _A1, respectively,
     * and the right-hand side vector b is stored ni @p _in_out.
     * 
     * @param _A3 Factorization of the matrix A3.
     * @param _A2 Factorization of the matrix A2.
     * @param _A1 Factorization of the matrix A1.
     * @param _in_out Right-hand side vector b and vector for storing the result x.
     * I.e., the initial information will be overwritten.
     */
    static void solveKronecker(const SymBandMatrixCholesky &_A3,
                                const SymBandMatrixCholesky &_A2,
                                const SymBandMatrixCholesky &_A1,
                                std::vector<real> &_in_out)
    {
        const auto n1 = _A1.n;
        const auto n2 = _A2.n;
        const auto n3 = _A3.n;

        assert(static_cast<int>(_in_out.size()) == (n1 * n2 * n3));

        real * const ptr_x = _in_out.data();

        for(int i = 0; i < n2 * n3; ++i)
        {
            _A1.forwardSubs(ptr_x + i * n1, 1);
            _A1.backwardSubs(ptr_x + i * n1, 1);
        }

        for(int j = 0; j < n3; ++j)
        {
            for(int i = 0; i < n1; ++i)
            {
                  _A2.forwardSubs(ptr_x + i + n1 * n2 * j, n1);
                  _A2.backwardSubs(ptr_x + i + n1 * n2 * j, n1);
            }
        }

        for(int i = 0; i < n1 * n2; ++i)
        {
            _A3.forwardSubs(ptr_x + i, n1 * n2);
            _A3.backwardSubs(ptr_x + i, n1 * n2);
        }
    }

private:

    /**
     * @brief Performs a forward substitution with the Cholesky factorization
     * for the a right-hand side vector initially stored in @p _in_out.
     * 
     * @param _in_out Right-hand side and solution vector.
     * I.e., the initial information will be overwritten.
     * @param _step Step to be used for moving from item of the right-hand
     * (or solution) vector to the next.
     */
    void forwardSubs(real *_in_out, const int _step) const
    {
        const auto &L = *this;
        for(int i = 0; i < n; ++i)
        {
            real rhs = _in_out[i * _step];

            const int start = std::max(0, i - this->bw);
            for(int j = start; j < i; ++j)
                rhs -= L(i, j) * _in_out[j * _step];

            _in_out[i * _step] = rhs / L(i, i);
        }
    }

    /**
     * @brief Performs a backward substitution with the Cholesky factorization
     * for the a right-hand side vector initially stored in @p _in_out.
     * 
     * @param _in_out Right-hand side and solution vector.
     * I.e., the initial information will be overwritten.
     * @param _step Step to be used for moving from item of the right-hand
     * (or solution) vector to the next.
     */
    void backwardSubs(real *_in_out, const int _step) const
    {
        const auto &L = *this;
        for(int i = n-1; i >= 0; --i)
        {
            real rhs = _in_out[i * _step];

            const int end = std::min(this->n, i + this->bw +1);
            for(int j = i + 1; j < end; ++j)
                rhs -= L(i, j) * _in_out[j * _step];

            _in_out[i * _step] = rhs / L(i, i);
        }
    }

    /**
     * @brief Performs the Cholesky factorization of the stored
     * symmetric band matrix assuming it is positive definite.
     */
    void factor()
    {
        for(int k = 0; k < n; ++k)
        {
            const int last = std::min(k+1+bw,n) - k;
            for(int j = 1; j < last; ++j)
            {
                const int i = k + j + 1;
                const auto fac = 1.0 / data[k];
                for(int l = 0; l < (last-j); ++l)
                    data[i-1 + n * l] -= fac * data[k + n * j] * data[k + n * (l+j)]; 
            }

            const auto fac = 1.0 / std::sqrt(data[k]);
            for (int j = 0; j < (bw + 1); ++j)
                 data[k + n * j] *= fac;
        }

        for(int k = n-bw; k < n; ++k)
            data[k + n * bw] = 0;
    }


};

inline std::ostream& operator<<(std::ostream& _os, const SymBandMatrix& _matrix)
{
    for(int i = 0; i < _matrix.n; ++i)
    {
        for(int j = 0; j < std::max(0, i-_matrix.bw); ++j)
            _os << 0.0 << " ";

        for(int j = std::max(0, i-_matrix.bw); j < std::min(_matrix.n, i + _matrix.bw +1) ; ++j)
            _os << _matrix(i, j) << " ";

        for(int j = std::min(_matrix.n, i + _matrix.bw +1); j < _matrix.n ; ++j)
            _os << 0.0 << " ";

        _os << std::endl;
    }
    return _os;
}



} // namespace algoim


#endif // ALGOIM_BAND_MATRIX_H