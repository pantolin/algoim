#ifndef ALGOIM_BSPLINE_H
#define ALGOIM_BSPLINE_H

#include "real.hpp"
#include "gaussquad.hpp"
#include "band_matrix.hpp"
#include "bezier.hpp"
#include "polynomial_tp.hpp"
#include "hyperrectangle.hpp"
#include "utility.hpp"
#include "uvector.hpp"
#include "multiloop.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>
#include <set>
#include <type_traits>
#include <vector>
#include <memory>

namespace algoim::bspline
{
namespace detail
{
    /**
     * @brief Transforms the given value of a real (double precision) one.
     * 
     * If the input type is a floating point type, the value is simply casted.
     * Instead, if the input value is an algoim interval, the alpha value
     * of the interval is returned.
     * Other cases are not implemented.
     * 
     * @tparam T Type of the input value.
     * @param _t Value to be transformed.
     * @return Transformed value.
     */
    template<typename T>
    real toReal(const T _t)
    {
        if constexpr (std::is_floating_point_v<T>)
            return static_cast<real>(_t);
        else
            return _t.alpha;
    }

} // namespace detail


struct BSpline;

/**
 * @brief Data structure for storing the evaluation of a 1D BSpline
 * space in a series of Gauss point in every element.
 * 
 * It actually stores the value of non-zero spline functions at every element
 * and at all the evaluation points, those value multiplied by the scaled
 * quadrature points, the coordinates of the evaluation points, and
 * the id of the first non-zero spline function for every element.
 */
struct BSplineEvaluation
{
    /**
     * @brief Constructor.
     * 
     * The constructor actually performs the evaluation.
     * 
     * @param _spline 1D BSpline to be evaluated.
     * @param _n_pts_per_elem Number of Gauss points per element.
     */
    BSplineEvaluation(const BSpline &_spline, const int _n_pts_per_elem);

    /// Order of the evaluated spline.
    const int order;
    /// Number of Gauss evaluation points per element.
    const int n_pts_per_elem;
    /// Number of elements.
    const int n_elems;
    /// Id of the first non-zero spline function for every element.
    std::vector<int> first_func_per_elem;

    /**
     * Values of the basis functions.
     * The values are stored for the non-zero basis functions,
     * at every element and at all the evaluation points.
     * 
     * The values are stored such that the iteration along the @ref order
     * non-zero functions, at an element and a precise point, is the fastest,
     * then, the iteration along the points, and finally the iteration along
     * the elements, that is the slowest.
     */
    std::vector<real> basis_values;

    /**
     * Same as @p basis_values, but multiplied by the quadrature weights.
     * The weights are already scaled with the measure of the element.
     */
    std::vector<real> basis_values_w;

    /**
     * Coordinates of the Gauss evaluation points.
     * 
     * The coordinates are stored such that the iteration along the points
     * within one element is the fastest, i.e., is the innermost,
     * and the iteration along the elements, that is the slowest.
     */
    std::vector<real> param_coords;

    /**
     * @brief Computes a mass matrix in the reference domain of the 1D Bspline.
     * 
     * @return The result is stored as a symmetric banded matrix.
     */
    std::shared_ptr<const SymBandMatrix> computeMass() const
    {
        assert(order <= n_pts_per_elem);
        const auto n_funcs = getNumFuncs();

        const auto mass_ptr = std::make_shared<SymBandMatrix>(n_funcs, (order - 1) * 2);   
        auto &mass = *mass_ptr;

        const auto b = basis_values.cbegin();
        const auto bw = basis_values_w.cbegin();

        for(int elem_id = 0; elem_id < n_elems; ++elem_id)
        {
            const auto f = this->first_func_per_elem[elem_id];

            for(int pt_id = 0; pt_id < n_pts_per_elem; ++pt_id)
            {
                const int offset = elem_id * n_pts_per_elem * order + pt_id * order;
                for(int i = 0; i < order; ++i)
                {
                    const auto bi = b + offset + i;
                    for(int j = i; j < order; ++j)
                    {
                        const auto bwj = bw + offset + j;
                        // const auto kk = std::inner_product(bi, bi + n_pts_per_elem, bwj, 0.0);
                        mass(f + i, f + j) += *bi * *bwj;
                    }
                }
            } // pt_id

        }

        return mass_ptr;
    }

    /**
     * @brief Returns the cardinality of the 1D Bspline space.
     * 
     * @return Number of function in the 1D BSpline space.
     */
    int getNumFuncs() const
    {
        return first_func_per_elem.back() + order;
    }

private:
    /**
     * @brief Computes the coordinates and weight of the Gauss-Legendre quadrature in [0, 1].
     * 
     * @param _n_pts Number of points.
     * @param _x Coordinates.
     * @param _w Weights.
     */
    static void computeGaussPoints(const int _n_pts, std::vector<real> &_x, std::vector<real> &_w)
    {
        assert(0 < _n_pts);
        _x.resize(_n_pts);
        _w.resize(_n_pts);
        for(int i = 0; i < _n_pts; ++i)
        {
            _x[i] = algoim::GaussQuad::x(_n_pts, i);
            _w[i] = algoim::GaussQuad::w(_n_pts, i);
        }
    }

    /**
     * @brief Method performing the evaluation of the spline space.
     * 
     * @param _bspline 1D BSpline space to be evaluated.
     */
    void init(const BSpline &_bspline);

};

/**
 * @brief Class for storing knots vector and related utilities.
 * 
 */
class Knots
{
public:
    /**
     * @brief Constructor.
     * @param _knots Non-decreasing sequence of knots.
     */
    Knots(const std::vector<real> &_knots)
    :
    data(_knots),
    unique_data()
    {
        // We assume that _knots is a non-decreasing sequence, but don't check.
        assert(1 < data.size());
        this->createUnique();
    }

    /**
     * @brief Constructor.
     * @param _unique_knots Unique sequence of knots.
     * @param _order Order of the spline.
     * @param _regularity Regularityr of the spline.
     */
    Knots(const std::vector<real> &_unique_knots,
          const int _order,
          const int _regularity)
    :
    data(),
    unique_data(_unique_knots)
    {
        // We assume that unique_knots is a non-decreasing sequence, but don't check.
        assert(1 < unique_data.size());
        assert(_regularity < (_order - 1));

        std::vector<int> n_reps(unique_data.size(), _order - 1 - _regularity);
        n_reps.front() = _order;
        n_reps.back() = _order;

        const int n_knots = std::accumulate(n_reps.cbegin(), n_reps.cend(), 0);
        data.reserve(n_knots);
        for(std::size_t i = 0; i < unique_data.size(); ++i)
        {
            for(int j = 0; j < n_reps[i]; ++j)
                data.push_back(unique_data[i]);
        }
    }

    /**
     * @brief Constructor for open uniform knot vector with maximum regularity,
     * for a given interval [ @p _k0, @p _k1 ].
     * 
     * @param _k0 First knot value.
     * @param _k1 Last knot value.
     * @param _n_spans Number of non-empty spans.
     * @param _order Order.
     */
    Knots(const real _k0, const real _k1, const int _n_spans, const int _order)
    :
    data(2 * _order + _n_spans - 1)
    {
        assert(_k0 < _k1);
        assert(0 < _n_spans);
        assert(0 < _order);

        auto it = data.begin();
        for(int i = 0; i < _order; ++i)
            *it++ = _k0;

        const real dx = (_k1 - _k0) / static_cast<real>(_n_spans);
        for(int i = 1; i < _n_spans; ++i)
            *it++ = _k0 + i * dx;

        for(int i = 0; i < _order; ++i)
            *it++ = _k1;

        this->createUnique();
    }

    /**
     * @brief Constructor for open uniform knot vector with maximum regularity.
     * 
     * @param _n_spans Number of non-empty spans.
     * @param _order Order.
     */
    Knots(const int _n_spans, const int _order)
    :
    Knots(0.0, 1.0, _n_spans, _order)
    {}

private:
    /**
     * @brief Extract the unique knots from the knots with repetitions
     * and stores them in @ref unique_data.
     */
    void createUnique()
    {
        constexpr real eps = std::numeric_limits<real>::epsilon() * 10.0;

        unique_data = data;

        unique_data.erase(std::unique(unique_data.begin(), unique_data.end(), [eps](const real &a, const real &b)
        {
            return std::fabs(a - b) < eps;
        }), unique_data.end());
    }

public:
    /**
     * @brief Refines the knot vector by inserting new knots.
     * @param _knots_to_insert New knots to insert (they may be repeated).
     */
    void refine(const std::vector<real> &_knots_to_insert)
    {
        for(const auto &knot : _knots_to_insert)
        {
            const auto i = this->getLastKnotSmallerOrEqual(knot);
            this->data.insert(this->data.begin() + i + 1, knot);
        }
        this->createUnique();
    }

    /**
     * @brief Refines (in place) the knot vector by subdividing knot spans.
     * @param _n_subdiv Number of spans in which every knot span must be subdivided.
     * @return New inserted knots.
     */
    std::vector<real> refine(const int _n_subdiv)
    {
        const auto n_spans = unique_data.size() - 1;

        std::vector<real> knots_to_insert;
        knots_to_insert.reserve(n_spans * (_n_subdiv - 1));

        assert(1 < _n_subdiv);
        for(std::size_t i = 0; i < n_spans; ++i)
        {
            const real k0 = unique_data[i];
            const real k1 = unique_data[i+1];
            const real dk = (k1 - k0) / _n_subdiv;
            for(int j = 1; j < _n_subdiv; ++j)
                knots_to_insert.push_back(k0 + dk * j);
        }

        this->refine(knots_to_insert);

        return knots_to_insert;
    }

    /**
     * @brief Updated the knot vector by raising its degree.
     * @param _old_order Old order.
     * @param _new_order New order.
     * @warning Only valid for open knot vectors.
     */
    void raise(const int _old_order, const int _new_order)
    {
        assert(isOpen(_old_order));

        const auto r = _new_order - _old_order;
        assert(0 < r);

        std::vector<real> knots_to_insert;
        knots_to_insert.reserve(unique_data.size() * r);
        for(const auto k : unique_data)
        {
            for(int i = 0; i < r; ++i)
                knots_to_insert.push_back(k);
        }

        this->refine(knots_to_insert);
    }

    /**
     * @brief Computes the multiplicity of the given @p _knot. It may be zero.
     * @param _knot Knot to be checked.
     * @return Multiplicity of @p _knot in the vector.
     */
    int getMultiplicity(const real _knot) const
    {
        constexpr real eps = std::numeric_limits<real>::epsilon() * 10.0;

        const int i = this->getLastKnotSmallerOrEqual(_knot);

        int m = 0;
        while (std::fabs(this->data[i - m] - _knot) <= eps && 0 <= (i - m))
        {
            ++m;
        }

        return m;
    }

    /**
     * @brief Splits the current knot vector into two vectors by inserting
     * the given @p _knot. It creates an open knot at the splitting point.
     * 
     * @param _order Order of the open knot vector.
     * @param _knot Knot for the split.
     * @return Generated knot vectors.
     */
    std::array<Knots, 2> splitAtKnot(const int _order, const real _knot) const
    {
        constexpr real eps = std::numeric_limits<real>::epsilon() * 10.0;
        assert(std::fabs(_knot - this->data.front()) > eps);
        assert(std::fabs(_knot - this->data.back()) > eps);

        const auto m = this->getMultiplicity(_knot);

        Knots ref_knots(this->data);
        ref_knots.refine(std::vector<real>(_order - m, _knot));

        const auto ind = ref_knots.getLastKnotSmallerOrEqual(_knot);
        const auto &ref_knots_data = ref_knots.data;

        const auto begin = ref_knots_data.cbegin();
        return {Knots(std::vector<real>(begin, begin + ind + 1)),
                Knots(std::vector<real>(begin + ind - _order + 1, ref_knots_data.cend()))};
    }

    /**
     * @brief Returns the unique knot values (up to a tolerance), i.e.,
     * without repetitions.
     * 
     * @return Knots without repetitions.
     */
    const std::vector<real> &getUnique() const
    {
        return unique_data;
    }

    /**
     * @brief Gets the number of non-empty knot spans.
     * 
     * @return Number of non-empty knot spans.
     */
    int getNumEelements() const
    {
        return static_cast<int>(unique_data.size()) - 1;
    }

    /**
     * @brief Returns the size of the knots vector (as an int).
     * 
     * @return Size of the knots vector.
     */
    int size() const
    {
        return static_cast<int>(this->data.size());
    }

    /**
     * @brief Checks whether the knot vector is an open knot vector of the given @p _order.
     * 
     * @param _order Order to be checked.
     * @return True if it is open, false otherwise.
     */
    bool isOpen(const int _order) const
    {
        return isOpen(this->data.data(), this->size(), _order);
    }

    /**
     * @brief Checks whether the knot vector is an open knot vector of the given @p _order.
     * 
     * @param _knots Pointer to the first entry of the knots vector.
     * @param _n_knots Total number of knots.
     * @param _order Order to be checked.
     * @return True if it is open, false otherwise.
     */
    static bool isOpen(const real * const _knots, const int _n_knots, const int _order)
    {
        assert(0 < _order);
        assert((2 * _order) <= _n_knots);

        constexpr real tolerance = std::numeric_limits<real>::epsilon() * 10.0;

        const auto k0 = _knots[0];
        for(int i = 1; i < _order; ++i)
        {
            if (std::fabs(_knots[i] - k0) > tolerance)
                return false;
        }

        const auto k1 = _knots[_n_knots -1];
        for(int i = _n_knots - _order; i < _n_knots; ++i)
        {
            if (std::fabs(_knots[i] - k1) > tolerance)
                return false;
        }

        return true;
    }

    /**
     * @brief Checks whether the knot vector is similar to the one of a Bezier.
     * I.e., it is an open knot vector with one single non-empty element.
     * 
     * @param _order Order to be checked.
     * @return True if it is open, false otherwise.
     */
    bool isBezierLike(const int _order) const
    {
        return isBezierLike(this->data.data(), this->size(), _order);
    }

    /**
     * @brief Checks whether the knot vector is similar to the one of a Bezier.
     * I.e., it is an open knot vector with one single non-empty element.
     * 
     * @param _knots Pointer to the first entry of the knots vector.
     * @param _n_knots Total number of knots.
     * @param _order Order to be checked.
     * @return True if it is open, false otherwise.
     */
    static bool isBezierLike(const real * const _knots, const int _n_knots, const int _order)
    {
        return isOpen(_knots, _n_knots, _order) && _n_knots == (2 * _order);
    }

    /**
     * @brief Gets the index of the knot that is smaller than or equal to @p _t
     * 
     * @tparam T Type of the coordinate.
     * @param _t Coordinate to be tested.
     * @return Index of the found knot value.
     */
    template<typename T>
    int getLastKnotSmallerOrEqual(const T _t) const
    {
        return getLastKnotSmallerOrEqual(this->data.data(), this->size(), _t);
    }

    /**
     * @brief Gets the index of the knot that is smaller than or equal to @p _t
     * 
     * @tparam T Type of the coordinate.
     * @param _knots Pointer to the first entry of the knots vector.
     * @param _n_knots Total number of knots.
     * @param _t Coordinate to be tested.
     * @return Index of the found knot value.
     */
    template<typename T>
    static int getLastKnotSmallerOrEqual(const real *_knots, const int _n_knots, const T _t)
    {
        constexpr real tolerance = std::numeric_limits<real>::epsilon() * 10.0;

        const auto t = detail::toReal(_t);
        const auto it = std::upper_bound(_knots, _knots + _n_knots, t + tolerance);
        return static_cast<int>(std::distance(_knots, it)) - 1;
    }

    /// Knots vector with repetitions.
    std::vector<real> data;

    /// Knots vector without repetitions.
    std::vector<real> unique_data;

};

namespace detail
{
    /**
     * @brief Refines a Bspline by inserting knot vectors.
     * 
     * @tparam N Tensor-product dimension of the Bspline.
     * @tparam T Control point type of the Bspline.
     * @param _dir Direction along which the refinement is performed.
     * @param _order Order along the refinement direction.
     * @param _old_pts Vector of control points of the original spline.
     * @param _n_old_pts Number of control points per direction of the original spline.
     * @param _old_knots Knots vector (with repetitions) of the original spline along the refinement direction.
     * @param _knots_to_insert List of knots to insert (it can contain) repetitions.
     * @param _new_pts Vector of control points of the refined spline.
     */
    template<int N, typename T>
    void refine(const int _dir, const int _order, const std::vector<T> &_old_pts, const uvector<int, N> &_n_old_pts, const Knots &_old_knots, const std::vector<real> &_knots_to_insert, std::vector<T> &_new_pts)
    {
        // Adaptation of the Algorithm 5.4 of The NURBS Book, Piegl & Tiller, 2nd edition.
        assert(0 < _order);
        assert(0 <= _dir && _dir < N);

        if (_knots_to_insert.empty())
            return;

        const auto n_knots_to_insert = static_cast<int>(_knots_to_insert.size());

        auto n_new_pts = _n_old_pts;
        n_new_pts(_dir) += n_knots_to_insert;

        assert(prod(_n_old_pts) == static_cast<int>(_old_pts.size()));
        assert(prod(n_new_pts) == static_cast<int>(_new_pts.size()));

        const auto create_multiloop = [n_new_pts, _dir](const int i)
        {
            uvector<int, N> k0(0), k1(n_new_pts);
            k0(_dir) = i;
            k1(_dir) = i + 1;
            return MultiLoop<N> (k0, k1);
        };

        const auto get_new_point = [&_new_pts, n_new_pts, _dir](const auto &i, const int offset = 0) -> T &
        {
            const auto ii = set_component(i(), _dir, i(_dir) + offset);
            return _new_pts[util::toFlatIndex<N>(n_new_pts, ii)];
        };

        const auto get_old_point = [&_old_pts, _n_old_pts, _dir](const auto &i, const int offset = 0) -> const T &
        {
            const auto ii = set_component(i(), _dir, i(_dir) + offset);
            return _old_pts[util::toFlatIndex<N>(_n_old_pts, ii)];
        };


        const auto degree = _order - 1;

        const auto span_0 = _old_knots.getLastKnotSmallerOrEqual(_knots_to_insert.front());
        const auto span_1 = _old_knots.getLastKnotSmallerOrEqual(_knots_to_insert.back());

        for (int j = 0; j <= span_0 - degree; ++j)
        {
            for (auto k = create_multiloop(j); ~k; ++k)
                get_new_point(k) = get_old_point(k);
        }

        for (int j = span_1; j < _n_old_pts(_dir); ++j)
        {
            for (auto k = create_multiloop(j); ~k;++k)
                get_new_point(k, n_knots_to_insert) = get_old_point(k);
        }

        Knots new_knots(_old_knots);
        new_knots.refine(_knots_to_insert);

        auto old_span = span_1 + degree;
        auto new_span = old_span + n_knots_to_insert;

        for (int j = n_knots_to_insert - 1; j >= 0; --j) {

            const auto new_knot = _knots_to_insert[j];

            while (new_knot <= _old_knots.data[old_span] && span_0 < old_span) {

                for (auto k = create_multiloop(new_span - _order); ~k; ++k)
                    get_new_point(k) = get_old_point(k, old_span - new_span);

                --new_span;
                --old_span;
            }

            for (auto k = create_multiloop(new_span - _order); ~k; ++k)
                get_new_point(k) = get_new_point(k, 1);

            for (int l = 1, new_ind = new_span - degree; l < _order; ++l, ++new_ind)  {

                constexpr auto alpha_tol = std::numeric_limits<real>::epsilon() * 10.0;
                real alpha = new_knots.data[new_span + l] - new_knot;
                if (std::fabs(alpha) < alpha_tol)    
                    alpha = 0.0;
                else
                    alpha /= new_knots.data[new_span + l] - _old_knots.data[old_span + l - degree];

                for (auto k = create_multiloop(new_ind); ~k; ++k)
                    get_new_point(k) = alpha * get_new_point(k) + (1.0 - alpha) * get_new_point(k, 1);
            }

            --new_span;
        }
    }

} // namespace detail


/**
 * @brief Class for representing 1D BSpline spaces and related utilities.
 */
struct BSpline
{
    /**
     * @brief Constructor.
     * 
     * @param _knots Knots vector (with repetitions).
     * @param _order Order to the space.
     */
    BSpline(const Knots &_knots, const int _order)
    :
    knots(_knots), order(_order)
    {
        assert(0 < order);
        assert(order <= getNumFunctions());
    }

    /**
     * @brief Constructor.
     * 
     * A space with equally sized elements and maximum regularity is created.
     * 
     * @param _n_funcs Cardinality of the space.
     * @param _order Order to the space.
     */
    BSpline(const int _n_funcs, const int _order)
    :
    knots(_n_funcs - _order + 1, _order), order(_order)
    {
        assert(0 < order);
    }

    /**
     * @brief Creator.
     * 
     * @param _knots Knots vector (with repetitions).
     * @param _order Order to the space.
     * @return Created space wrapped in a shared pointer.
     */
    static std::shared_ptr<BSpline>
    create(const Knots &_knots, const int _order)
    {
        return std::make_shared<BSpline>(_knots, _order);
    }

    /// Knots vector of the spline space.
    Knots knots;
    /// Order of the spline space.
    const int order;

    /**
     * @brief Creates a new refined BSpline by inserting knots.
     * @param _knots_to_insert Knots to insert (they may be repeated).
     * @return New generated BSpline.
     */
    std::shared_ptr<BSpline> refine(const std::vector<real> &_knots_to_insert) const
    {
        const auto new_bspline = std::make_shared<BSpline>(this->knots, order);
        new_bspline->knots.refine(_knots_to_insert);
        return new_bspline;
    }

    /**
     * @brief Creates a new BSpline by raising its degree.
     * @param _new_order Order of the splien after raising it.
     * @return New generated BSpline.
     */
    std::shared_ptr<BSpline> raise(const int _new_order) const
    {
        assert(order < _new_order);
        Knots new_knots(this->knots);
        new_knots.raise(order, _new_order);
        const auto new_bspline = std::make_shared<BSpline>(new_knots, _new_order);
        return new_bspline;
    }

    /**
     * @brief Creates a new refined BSpline by subdividing knot spans.
     * @param _n_subdiv Number of spans in which every knot span must be subdivided.
     * @return New generated BSpline.
     */
    std::shared_ptr<BSpline> refine(const int _n_subdiv) const
    {
        const auto new_bspline = std::make_shared<BSpline>(this->knots, order);
        new_bspline->knots.refine(_n_subdiv);
        return new_bspline;
    }

    /**
     * @brief Returns the number of functions in the space.
     * 
     * @return Space cardinality.
     */
    int getNumFunctions() const
    {
        return getNumFunctions(this->knots, this->order);
    }

    /**
     * @brief Returns the number of functions in the space.
     * 
     * @param _knots Knots vector (with repetitions).
     * @param _order Order to the space.
     * @return Space cardinality.
     */
    static int getNumFunctions(const Knots &_knots, const int _order)
    {
        return _knots.size() - _order;
    }

    /**
     * @brief Creates a @ref BSplineEvalution of the current spline space.
     * 
     * @param _n_pts_per_elem Number of Gauss points per element.
     * @return Evaluation of the space.
     */
    std::shared_ptr<const BSplineEvaluation> evaluate(const int _n_pts_per_elem) const
    {
        return std::make_shared<BSplineEvaluation>(*this, _n_pts_per_elem);  
    }

    /**
     * @brief Computes the index of the first non-zero spline function
     * at the coordinate @p _t.
     * 
     * @tparam T Type of the coordinate.
     * @param _t Coordinate at which the first index is computed.
     * @return Index of the first non-zero spline function at @p _t.
     */
    template<typename T>
    int getFirstBasisIndex(const T &_t) const
    {
        return getFirstBasisIndex(knots.data.data(), knots.size(), order, _t);
    }

    /**
     * @brief Computes the index of the first non-zero spline function
     * at the coordinate @p _t.
     * 
     * @tparam T Type of the coordinate.
     * @param _knots Knots vector (with repetitions).
     * @param _n_knots Total number of knots.
     * @param _order Order to the space.
     * @param _t Coordinate at which the first index is computed.
     * @return Index of the first non-zero spline function at @p _t.
     */
    template<typename T>
    static int getFirstBasisIndex(const real * const _knots, const int _n_knots, const int _order, const T &_t)
    {
        const int id = Knots::getLastKnotSmallerOrEqual(_knots, _n_knots, _t);
        if (id >= (_n_knots - 1))
            return _n_knots - 2 * _order;
        else
            return id - _order + 1;
    }


    /**
     * @brief Evaluates the non-zero spline functions at the coordinate @p _t.
     * 
     * @tparam T Type of the coordinate.
     * @param _knots Knots vector (with repetitions).
     * @param _n_knots Total number of knots.
     * @param _order Order to the space.
     * @param _t Coordinate at which the evaluation is computed.
     * @param _values Values of the non-zero spline functions at @p _t.
     */
    template<typename T>
    static void evaluateBasis(const real * const _knots, const int _n_knots, const int _order, const T &_t, T *_values)
    {
        assert(0 < _order);
        assert(2 * _order <= _n_knots);
        assert(Knots::isOpen(_knots, _n_knots, _order));

        constexpr real tolerance = std::numeric_limits<real>::epsilon() * 10.0;

        // Important note: This algorithm has been partially copied from IRIT, developed by Gershon Elber.

        const int n_funcs = _n_knots - _order;

        for(int i = 0; i < _order; ++i)
            _values[i] = T(0.0);

        const auto t = detail::toReal(_t);
        if (t <= (_knots[0] + tolerance))
        {
            _values[0] = T(1.0);
            return;
        }
        else if ((_knots[_n_knots - 1] - tolerance) <= t)
        {
            _values[_order-1] = T(1.0);
            return;
        }

        if (Knots::isBezierLike(_knots, _n_knots, _order))
        {
            const auto t_01 = (_t - _knots[0]) / (_knots[_n_knots - 1] - _knots[0]);
            bezier::evaluateBernstein(t_01, _order, 0, _values);
            return;
        }

        constexpr real COX_DB_IRIT_EPS = 1.e-20;

        
        const int first_knot = Knots::getLastKnotSmallerOrEqual(_knots, _n_knots, _t);

        _values[0] = T(1.0);

        /* Here is the tricky part. we raise these constant geometry to the               */
        /* required order of the curve Crv for the given parameter _t. There are           */
        /* at most order non zero function at param. value _t. These functions             */
        /* start at index-order+1 up to index (order functions over whole).               */
        for (int i = 2; i <= _order; ++i) { /* Goes through all orders...                  */
            /* This code is highly optimized from the commented out code below.           */
            /* for (int l = i - 1; l >= 0; l--) {                                         */
            /*  s1 = (KnotVector[first_knot + l] - KnotVector[first_knot + l - i + 1]);   */
            /*  s1 = COX_DB_IRIT_APX_EQ(s1, 0.0)				                          */
            /*	? 0.0 : (_t - KnotVector[first_knot + l - i + 1]) / s1;	                  */
            /*  s2 = (KnotVector[first_knot + l + 1]-KnotVector[first_knot + l - i + 2]); */
            /*  s2 = COX_DB_IRIT_APX_EQ(s2, 0.0)				                          */
            /*	? 0.0 : (KnotVector[first_knot + l + 1] - _t) / s2;		                  */
            /*  _values[l] = s1 * _values[l - 1] + s2 * _values[l];	                      */
            const real *KVIndexl   = _knots + first_knot + i - 1; /* KV[first_knot + l]         */
            const real *KVIndexl1  = KVIndexl + 1;                      /* KV[first_knot + l + 1]     */
            const real *KVIndexli1 = KVIndexl - i + 1;                  /* KV[first_knot + l - i + 1] */
            real s1, s2, s2inv;
            T *BF = _values + i - 1;

            if ((s2 = *KVIndexl1 - KVIndexli1[1]) >= COX_DB_IRIT_EPS)
                s2inv = 1.0 / s2;
            else
                s2inv = 0.0;

            for (int l = i - 1; l >= 0; --l) { /* And all basis funcs. of order i. */
                if (s2inv == 0.0) {
                    *BF-- = T(0.0);
                    KVIndexl1--;
                }
                else
                    *BF-- *= (*KVIndexl1-- - _t) * s2inv;

                if (l > 0 && (s1 = *KVIndexl-- - *KVIndexli1--) >= COX_DB_IRIT_EPS) {
                    s2inv = 1.0 / s1;
                    BF[1] += BF[0] * (_t - KVIndexli1[1]) * s2inv;
                }
                else
                    s2inv = 0.0;
            }
        }

        return;
    }

    /**
     * @brief Evaluates the derivatives of the non-zero spline functions at the coordinate @p _t.
     * 
     * @tparam T Type of the coordinate.
     * @param _knots Knots vector (with repetitions).
     * @param _n_knots Total number of knots.
     * @param _order Order to the space.
     * @param _t Coordinate at which the evaluation is computed.
     * @param _values Deritative values of the non-zero spline functions at @p _t.
     * @param _der Derivative order (it must be non-negative).
     */
    template<typename T>
    static void evaluateBasisDerivative(const real * const _knots, const int _n_knots, const int _order, const T &_t, T *_values, const int _der)
    {
        assert(0 < _order);
        assert(0 <= _der);
        assert(2 * _order <= _n_knots);
        assert(Knots::isOpen(_knots, _n_knots, _order));

        if (_order <= _der)
        {
            for(int i = 0; i < _order; ++i)
                _values[i] = T(0.0);
        }
        else if (_der == 0)
        {
            evaluateBasis(_knots, _n_knots, _order, _t, _values);
        }
        else // if (0 < _der)
        {
            // Evaluating derivative one order less.
            std::vector<T> der_values(_order - 1);
            evaluateBasisDerivative(_knots + 1, _n_knots - 2, _order - 1, _t, der_values.data(), _der - 1);

            const auto first_id_low = getFirstBasisIndex(_knots + 1, _n_knots - 2, _order - 1, _t);

            const int degree = _order - 1;

            const auto first_id = getFirstBasisIndex(_knots, _n_knots, _order, _t);
            for(int i = 0; i < _order; ++i)
            {
                const auto ii = first_id + i;
                const auto ii_low = ii - first_id_low - 1;

                _values[i] = T(0.0);
		        if (0 <= ii_low && ii_low < degree)
		            _values[i] += der_values[ii_low] / (_knots[ii + degree] - _knots[ii]);
    		    if (0 <= (ii_low + 1) && (ii_low + 1) < degree)
		            _values[i] -= der_values[ii_low + 1] / (_knots[ii + _order] - _knots[ii + 1]);
                _values[i] *= degree;
            }
        }
    }

    /**
     * @brief Evaluates the derivatives of the non-zero spline functions at the coordinate @p _t.
     * 
     * @tparam T Type of the coordinate.
     * @param _t Coordinate to be tested.
     * @param _values Deritative values of the non-zero spline functions at @p _t.
     * @param _der Derivative order (it must be non-negative).
     */
    template<typename T>
    void evaluateBasisDerivative(const T &_t, T *_values, int _der) const
    {
        evaluateBasisDerivative(knots.data.data(), knots.size(), order, _t, _values, _der);
    }

    /**
     * @brief Evaluates the non-zero spline functions at the coordinate @p _t.
     * 
     * @tparam T Type of the coordinate.
     * @param _t Coordinate to be tested.
     * @param _values Values of the non-zero spline functions at @p _t.
     */
    template<typename T>
    void evaluateBasis(const T &_t, T *_values) const
    {
        this->evaluateBasisDerivative(_t, _values, 0);
    }
};

inline BSplineEvaluation::BSplineEvaluation(const BSpline &_spline, const int _n_pts_per_elem)
:
order(_spline.order),
n_pts_per_elem(_n_pts_per_elem),
n_elems(_spline.knots.getNumEelements()),
first_func_per_elem(),
basis_values(),
basis_values_w(),
param_coords()
{
    assert(0 < order);
    assert(0 < n_pts_per_elem);
    assert(0 < n_elems);

    this->init(_spline);
}

inline void BSplineEvaluation::init(const BSpline &bspline)
{
    first_func_per_elem.resize(n_elems);
    basis_values.resize(order * n_elems * n_pts_per_elem);
    basis_values_w.resize(order * n_elems * n_pts_per_elem);
    param_coords.resize(n_elems * n_pts_per_elem);

    const auto &unique_knots = bspline.knots.getUnique();

    std::vector<real> xq, wq;
    computeGaussPoints(n_pts_per_elem, xq, wq);

    auto *v = this->basis_values.data();
    auto *vw = this->basis_values_w.data();
    auto *c = this->param_coords.data();
    auto *f = this->first_func_per_elem.data();

    for(int elem_id = 0; elem_id < n_elems; ++elem_id)
    {
        const auto u0 = unique_knots[elem_id];
        const auto u1 = unique_knots[elem_id + 1];

        for(int pt_id = 0; pt_id < n_pts_per_elem; ++pt_id)
        {
            *c = u0 + (u1 - u0) * xq[pt_id];
            const auto w = (u1 - u0) * wq[pt_id];

            bspline.evaluateBasis(*c, v);
            for (int i = 0; i < order; ++i)
                vw[i] = v[i] * w;

            v += order;
            vw += order;
            ++c;
        }

        *f++ = bspline.getFirstBasisIndex(c[-1]);

    } // elem_id
}

/**
 * @brief N-dimensional tensor-product BSpline function.
 * 
 * @tparam N Number of dimensions.
 * @tparam T Base type of the coefficient variables.
 * @tparam N Dimension of the parametric domain.
 * @tparam R Dimension of the image.
 */
template<int N, int R, typename T = real>
struct BSplineTP
{
    /// Coefs type.
    using CoefsType = std::conditional_t<R == 1, T, uvector<T, R>>;

    /**
     * @brief Type of the value obtained with when the operator() is called.
     * @tparam T_ Type of the input coordinates.
     */
    template <typename T_>
    using Val = typename PolynomialTP<N, R, T>::template Val<T_>;

    /**
     * @brief Type of the value obtained with when the grad() is called.
     * @tparam T_ Type of the input coordinates.
     */
    template <typename T_>
    using Grad = typename PolynomialTP<N, R, T>::template Grad<T_>;

    /**
     * @brief Type of the value obtained with when the hessian() is called.
     * @tparam T_ Type of the input coordinates.
     */
    template <typename T_>
    using Hess = typename PolynomialTP<N, R, T>::template Hess<T_>;

    /**
     * Values of the spline control points.
     * The ordering is such that dimension N-1 is inner-most, i.e.,
     * iterates the fastest, while dimension 0 is outer-most and iterates the slowest.
     */
    std::vector<CoefsType> coefs;
    /** 1D BSpline spaces along the N directions. */
    std::array<std::shared_ptr<const BSpline>, N> bsplines_1D;
    /// N-dimensional array view of the coefficients.
    xarray<CoefsType, N> coefs_xarray;

    /**
     * @brief Construct a new tensor-product BSpline function.
     * 
     * @param _coefs Values of the spline control points.
     * The ordering is such that dimension N-1 is inner-most, i.e.,
     * iterates the fastest, while dimension 0 is outer-most and iterates the slowest.
     * @param _bsplines_1D 1D BSpline spaces along the N directions.
     */
    BSplineTP(const std::array<std::shared_ptr<const BSpline>, N> &_bsplines_1D)
              :
              coefs(prod(getNumPtsDir(_bsplines_1D))),
              bsplines_1D(_bsplines_1D),
              coefs_xarray(this->coefs.data(), getNumPtsDir(_bsplines_1D))
    {}

    /**
     * @brief Construct a new tensor-product BSpline function.
     * 
     * @param _coefs Values of the spline control points.
     * The ordering is such that dimension N-1 is inner-most, i.e.,
     * iterates the fastest, while dimension 0 is outer-most and iterates the slowest.
     * @param _bsplines_1D 1D BSpline spaces along the N directions.
     */
    BSplineTP(const std::vector<CoefsType> &_coefs,
              const std::array<std::shared_ptr<const BSpline>, N> &_bsplines_1D)
              :
              coefs(_coefs),
              bsplines_1D(_bsplines_1D),
              coefs_xarray(this->coefs.data(), getNumPtsDir(_bsplines_1D))
    {
        assert(static_cast<int>(coefs.size()) == prod(this->getNumPtsDir()));
    }

    /**
     * @brief Copy constructor
     * 
     * @param _bspline TP BSpline to be copied.
     */
    BSplineTP(const BSplineTP<N, R, T> &_bspline)
              :
              coefs(_bspline.coefs),
              bsplines_1D(_bspline.bsplines_1D),
              coefs_xarray(this->coefs.data(), getNumPtsDir(_bspline.bsplines_1D))
    {}

    /**
     * @brief Constructor from Bezier.
     * 
     * @param _bezier TP Bezier to be transformed.
     */
    BSplineTP(const bezier::BezierTP<N, R, T> &_bezier)
              :
              coefs(_bezier.coefs),
              bsplines_1D(),
              coefs_xarray(this->coefs.data(), _bezier.order)
    {
        for(int dir = 0; dir < N; ++dir)
        {
            const auto order = _bezier.order(dir);
            bsplines_1D[dir] = std::make_shared<BSpline>(order, order);
        }
    }

    /**
     * @brief Creator.
     * 
     * @param _coefs Values of the spline control points.
     * The ordering is such that dimension N-1 is inner-most, i.e.,
     * iterates the fastest, while dimension 0 is outer-most and iterates the slowest.
     * @param _bsplines_1D_vec 1D BSpline spaces along the N directions.
     * @return Created bspline wrapped in a shared pointer.
     *
     * @note This static function exists for being called from the Python wrapper.
     * @note The reason why @p _bsplines_1D_vec is a std::vector instead of a std::array
     * is due to limitations with cppyy library.
     */
    static std::shared_ptr<BSplineTP>
    create(const std::vector<CoefsType> &_coefs,
           const std::vector<std::shared_ptr<const BSpline>> &_bsplines_1D_vec)
    {
        static_assert(std::is_same_v<T, real>, "Only implmented for real.");

        assert(_bsplines_1D_vec.size() == N);
        std::array<std::shared_ptr<const BSpline>, N> bsplines_1D;
        for(int dir = 0; dir < N; ++dir)
            bsplines_1D[dir] = _bsplines_1D_vec[dir];

        return std::make_shared<BSplineTP>(_coefs, bsplines_1D);
    }

    /**
     * @brief Creates a copy and resturns it wrapped in a shared pointer.
     *
     * The copy is only performed for the coefficients, not the 1D spaces.
     * 
     * @return Copy wrapped in a shared pointer.
     */
    std::shared_ptr<BSplineTP> clone() const
    {
        return std::make_shared<BSplineTP>(this->coefs, this->bsplines_1D);
    }

    /**
     * @brief Gets the dimension of the spline.
     * @return Dimension of the spline.
     */
    static int getDim()
    {
        return N;
    }

    /**
     * @brief Gets the range of the spline.
     * @return Range of the spline.
     */
    static int getRange()
    {
        return R;
    }

    /**
     * @brief Gets a constant reference to the stored xarray view of the control point coefficients.
     * @return Constant view of the control point coefficients.
     */
    const xarray<CoefsType, N> &getXarray() const
    {
        return coefs_xarray;
    }

    /**
     * @brief Constructs a new N-dimensional tensor-product BSpline function by
     * computing a L2 projection of the function @p _func and
     * returns it wrapped in a shared pointer.
     * 
     * The created spline has uniform open knots vectors and maximum regularity.
     * 
     * @tparam F Function type.
     * @param _func Function to be projected.
     * @param _domain Considered function's domain.
     * @param _order Order of the BSpline along the N dimensions.
     *               All the orders must be greater than zero.
     * @param _n_elems Number of elements per direction.
     * @param _n_sample_pts_per_elem Number of Gauss points per element for
     * computing the L2 projection.
     * 
     * @return Created spline wrapped in shared pointer.
     * 
     * @note Currently, this method is only supported for real-valued functions.
     */
    template<typename F, int R_ = R>
    static std::enable_if_t<R_ == 1, std::shared_ptr<BSplineTP<N,R>>>
    L2Projection(const F &_func,
                  const HyperRectangle<real, N> &_domain,
                  const uvector<int, N> &_order,
                  const uvector<int, N> &_n_elems,
                  const uvector<int, N> &_n_sample_pts_per_elem)
    {
        std::array<std::shared_ptr<const BSpline>, N> bsplines_1D;
        const auto create_bspline = [&](const int dir)
        {
            const Knots knots(_domain.min(dir), _domain.max(dir), _n_elems(dir), _order(dir));
            bsplines_1D[dir] = std::make_shared<BSpline>(knots, _order(dir));
        };

        create_bspline(0);

        for(int dir = 1; dir < N; ++dir)
        {
            if (_order(dir) == _order(dir-1))
                bsplines_1D[dir] = bsplines_1D[dir-1];
            else
                create_bspline(dir);
        }

        const auto bspline = std::make_shared<BSplineTP<N, R>>(bsplines_1D);
        bspline->projectL2(_func, _n_sample_pts_per_elem);

        return bspline;
    }

    /**
     * @brief Gets the number of control points per direction of the given 1D BSplines.
     * @param _bsplines_1D 1D Bspline whose number of control points are extracted.
     * @return Number of control points per direction.
     */
    static uvector<int, N> getNumPtsDir(const std::array<std::shared_ptr<const BSpline>, N> &_bsplines_1D)
    {
        uvector<int, N> n;
        for(int dir = 0; dir < N; ++dir)
            n(dir) = _bsplines_1D[dir]->getNumFunctions();
        return n;
    }

    /**
     * @brief Gets the number of control points per direction.
     * @return Number of control points per direction.
     */
    uvector<int, N> getNumPtsDir() const
    {
        return getNumPtsDir(this->bsplines_1D);
    }

    /**
     * @brief Gets the order along each parametric direction.
     * @return Order along each direction.
     */
    uvector<int, N> getOrderDir() const
    {
        uvector<int, N> order;
        for(int dir = 0; dir < N; ++dir)
            order(dir) = bsplines_1D[dir]->order;
        return order;
    }

    /**
     * @brief Gets the number of elements (non-zero knot spans) per direction.
     * @return Number of elements per direction.
     */
    uvector<int, N> getNumElemsDir() const
    {
        uvector<int, N> n;
        for(int dir = 0; dir < N; ++dir)
            n(dir) = this->getKnots(dir).getNumEelements();
        return n;
    }

    /**
     * @brief Gets the nots vector along direction @p _dir.
     * @return Constant reference to knots vector.
     */
    const Knots &getKnots(const int _dir) const
    {
        assert(0 <= _dir && _dir < N);
        return bsplines_1D[_dir]->knots;
    }

    /**
     * @brief Gets the local domains (bounding boxes) of all
     * the elements in the spline parametric domain.
     * 
     * @return Vector containing the local domains of every element.
     * The ordering of the elements is the same as the one of the
     * control points.
     * I.e., the ordering is such that dimension N-1 is inner-most, then,
     * iterates the fastest, while dimension 0 is outer-most and iterates
     * the slowest.
     */
    std::vector<HyperRectangle<real, N>> getElementsDomains() const
    {
        const auto n_elems_dir = this->getNumElemsDir();

        std::vector<HyperRectangle<real, N>> elem_domains;
        elem_domains.reserve(prod(n_elems_dir));

        for(MultiLoop<N> i(0, n_elems_dir); ~i; ++i)
        {
            HyperRectangle<real, N> elem_domain(0, 1);
            for(int dir = 0; dir < N; ++dir)
            {
                const auto &unique_knots = this->getKnots(dir).getUnique();
                elem_domain.min(dir) = unique_knots[i(dir)];
                elem_domain.max(dir) = unique_knots[i(dir) + 1];
            }
            elem_domains.push_back(elem_domain);
        }

        return elem_domains;
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

    /**
     * @brief Gets the element to which a point belongs to.
     * 
     * @param _param_pt Point to be checked.
     * @return Tensor index of the element.
     */
    uvector<int, N> findElement(const uvector<real, N> &_param_pt) const
    {
        const auto n_elems_dir = this->getNumElemsDir();

        uvector<int, N> tid;
        for(int dir = 0; dir < N; ++dir)
        {
            const auto &unique_knots = this->getKnots(dir).getUnique();
            const auto it = std::upper_bound(unique_knots.cbegin(), unique_knots.cend(), _param_pt(dir));
            tid(dir) = static_cast<int>(std::distance(unique_knots.begin(), std::prev(it)));
            tid(dir) = std::min(tid(dir), static_cast<int>(unique_knots.size() - 2));
        }

        return tid;
    }

    /**
     * @brief Gets the parametric domain of the given element.
     * @param _tid Tensor index of the element.
     * @return Parametric domain of the element.
     */
    HyperRectangle<T, N> getElementDomain(const uvector<int, N> &_tid) const
    {
        HyperRectangle<T, N> domain(0, 1);

        for(int dir = 0; dir < N; ++dir)
        {
            const auto &unique_knots = this->getKnots(dir).getUnique();
            assert(0 <= _tid(dir) && _tid(dir) < (unique_knots.size() - 1));

            domain.min(dir) = unique_knots[_tid(dir)];
            domain.max(dir) = unique_knots[_tid(dir) + 1];
        }

        return domain;
    }

    /**
     * @brief Returns the element flat index associated to
     * the element tensor index.
     * 
     * @param _tid Element tensor index.
     * @return Element flat index.
     */
    int getFlatIndex(const uvector<int, N> &_tid) const
    {
        const auto n_elems = this->getNumElemsDir();
        return util::toFlatIndex<N>(n_elems, _tid);
    }

    /**
     * @brief Returns the element tensor index associated to
     * the element flat index.
     * 
     * @param _id Element flat index.
     * @return Element tensor index.
     */
    uvector<int, N> getTensorIndex(int _id) const
    {
        const auto n_elems = this->getNumElemsDir();
        return util::toTensorIndex<N>(n_elems, _id);
    }

    /**
     * @brief Checks if the current BSpline can be transformed into a single Bezier.
     * 
     * @return True if it can be transformed into a Bezier, false otherwise.
     */
    bool isBezierLike() const
    {
        const auto order = this->getOrderDir();
        for(int dir = 0; dir < N; ++dir)
        {
            if (!this->getKnots(dir).isBezierLike(order(dir)))
                return false;
        }
        return true;
    }

    /**
     * @brief Gets the domain of the BSpline space.
     * 
     * @return Domain of the space.
     */
    HyperRectangle<real, N> getDomain() const
    {
        HyperRectangle<real, N> domain(0, 1);

        for(int dir = 0; dir < N; ++dir)
        {
            const auto &knots = this->getKnots(dir);
            domain.min(dir) = knots.data.front();
            domain.max(dir) = knots.data.back();
        }

        return domain;
    }
    
    /**
     * @brief Transforms the current BSpline into a Bezier.
     * 
     * @note Along each direction, the BSpline must have an
     * open knot vector with a single non-zero span.
     * 
     * @return Created Bezier wrapped in a shared pointer.
     */
    std::shared_ptr<bezier::BezierTP<N, R, T>> transformToBezier() const
    {
        assert(this->isBezierLike());
        const auto order = this->getOrderDir();
        return std::make_shared<bezier::BezierTP<N, R, T>>(this->coefs, order);
    }

    /**
     * @brief Refines the current spline inserting new knots.
     * 
     * @param _ref_dir Direction along which the refinement is performed.
     * @param _knots_to_insert List of knots to insert (it can contain) repetitions.
     * @return Newly generated refined spline wrapped in a shared pointer.
     */
    std::shared_ptr<BSplineTP<N, R, T>> refine(const int _ref_dir, const std::vector<real> &_knots_to_insert) const
    {
        assert(0 <= _ref_dir && _ref_dir < N);

        if (_knots_to_insert.empty())
        {
            return std::make_shared<BSplineTP<N, R, T>>(*this);
        }

        auto new_bsplines_1D = bsplines_1D;
        new_bsplines_1D[_ref_dir] = new_bsplines_1D[_ref_dir]->refine(_knots_to_insert);

        const auto new_bspline = std::make_shared<BSplineTP<N, R, T>>(new_bsplines_1D);

        detail::refine(_ref_dir, this->getOrderDir()(_ref_dir), this->coefs, this->getNumPtsDir(), this->getKnots(_ref_dir), _knots_to_insert, new_bspline->coefs);

        return new_bspline;
    }

    /**
     * @brief Refines the current spline subdiving the knot spans.
     * 
     * @param _ref_dir Direction along which the refinement is performed.
     * @param _n_subdiv Number of spans in which every knot span must be subdivided.
     * @return Newly generated refined spline wrapped in a shared pointer.
     */
    std::shared_ptr<BSplineTP<N, R, T>> refine(const int _ref_dir, const int _n_subdiv) const
    {
        assert(0 <= _ref_dir && _ref_dir < N);

        if (_n_subdiv < 2)
        {
            return std::make_shared<BSplineTP<N, R, T>>(*this);
        }

        auto new_bsplines_1D = bsplines_1D;
        Knots new_knots(new_bsplines_1D[_ref_dir]->knots);
        const auto knots_to_insert = new_knots.refine(_n_subdiv);

        new_bsplines_1D[_ref_dir] = std::make_shared<BSpline>(new_knots, this->getOrderDir()(_ref_dir));

        const auto new_bspline = std::make_shared<BSplineTP<N, R, T>>(new_bsplines_1D);

        detail::refine(_ref_dir, this->getOrderDir()(_ref_dir), this->coefs, this->getNumPtsDir(), this->getKnots(_ref_dir), knots_to_insert, new_bspline->coefs);

        return new_bspline;
    }

    /**
     * @brief Refines the current spline subdiving the knot spans.
     * 
     * @param _n_subdiv Number of spans in which every knot span must be subdivided, for every direction.
     * @return Newly generated refined spline wrapped in a shared pointer.
     */
    std::shared_ptr<BSplineTP<N, R, T>> refine(const uvector<int, N> &_n_subdiv) const
    {
        std::shared_ptr<BSplineTP<N, R, T>> ref{nullptr};
        for(int dir = 0; dir < N; ++dir)
        {
            assert(0 < _n_subdiv(dir));
            if (_n_subdiv(dir) == 1)
                continue;

            if (ref == nullptr)
                ref = this->refine(dir, _n_subdiv(dir));
            else
                ref = ref->refine(dir, _n_subdiv(dir));
        }

        if (ref == nullptr)
            ref = this->clone();

        return ref;
    }

    /**
     * @brief Splits the current spline into two new ones along
     * the parametric direction with a higher number of knots.
     * In addition, the split is performed at the middle knot (approx)
     * already present in the knot vector.
     * 
     * @return Two generated spline (on the left and right of the splitting knot).
     */
    std::array<std::shared_ptr<BSplineTP<N, R, T>>, 2> split() const
    {
        assert(!this->isBezierLike());

        const auto n_elems = this->getNumElemsDir();
        const auto dir = argmax(n_elems);

        const auto &unique_knots = this->getKnots(dir).getUnique();
        const auto id = (static_cast<int>(unique_knots.size()) - 1) / 2;
        const auto knot = unique_knots[id];

        return splitAtKnot(dir, knot);
    }

    /**
     * @brief Splits the current spline into two new ones at a certain knot.
     * 
     * @param _ref_dir Direction along which the split is performed.
     * @param _knot Knot at which the split is performed.
     * @return Two generated spline (on the left and right of the @p _knot).
     */
    std::array<std::shared_ptr<BSplineTP<N, R, T>>, 2> splitAtKnot(const int _ref_dir, const real _knot) const
    {
        assert(0 <= _ref_dir && _ref_dir < N);

        const auto &knots = this->getKnots(_ref_dir);
        const auto order = this->getOrderDir()(_ref_dir);

        const auto new_knots = knots.splitAtKnot(order, _knot);

        auto bsplines_1D_0 = bsplines_1D;
        auto bsplines_1D_1 = bsplines_1D;
        bsplines_1D_0[_ref_dir] = std::make_shared<BSpline>(new_knots[0], order);
        bsplines_1D_1[_ref_dir] = std::make_shared<BSpline>(new_knots[1], order);

        const auto bspline_0 = std::make_shared<BSplineTP<N, R, T>>(bsplines_1D_0);
        const auto bspline_1 = std::make_shared<BSplineTP<N, R, T>>(bsplines_1D_1);


        const auto m = knots.getMultiplicity(_knot);
        const std::vector<real> knots_to_insert(order - 1 - m, _knot);
        const auto ref_spline = this->refine(_ref_dir, knots_to_insert);


        const auto n_pts = ref_spline->getNumPtsDir();
        const auto n_pts_0 = bspline_0->getNumPtsDir();
        const auto n_pts_1 = bspline_1->getNumPtsDir();

        const int brk_pt = n_pts_0(_ref_dir) - 1;

        uvector<int, N> i0(0), i1 = n_pts;

        const auto &coefs_ref = ref_spline->coefs;
        auto &coefs_0 = bspline_0->coefs;

        for(int j = 0; j <= brk_pt; ++j)
        {
            i0(_ref_dir) = j;
            i1(_ref_dir) = j + 1;

            for(MultiLoop<N> i(i0, i1); ~i; ++i)
            {
                const auto id0 = util::toFlatIndex<N>(n_pts_0, i());
                const auto id = util::toFlatIndex<N>(n_pts, i());
                coefs_0[id0] = coefs_ref[id];
            }
        }

        auto &coefs_1 = bspline_1->coefs;
        for(int j = brk_pt; j < n_pts(_ref_dir); ++j)
        {
            i0(_ref_dir) = j;
            i1(_ref_dir) = j + 1;

            for(MultiLoop<N> i(i0, i1); ~i; ++i)
            {
                const auto id1 = util::toFlatIndex<N>(n_pts_1, set_component(i(), _ref_dir, j-brk_pt));
                const auto id = util::toFlatIndex<N>(n_pts, i());
                coefs_1[id1] = coefs_ref[id];
            }
        }

        return {bspline_0, bspline_1};
    }

    /**
     * @brief Splits the BSpline function into a collection of Beziers.
     * 
     * @return Collection of generated Beziers.
     * Their ordering is such that dimension N-1 is inner-most, i.e.,
     * iterates the fastest, while dimension 0 is outer-most and iterates
     * the slowest.
     */
    std::vector<std::shared_ptr<bezier::BezierTP<N, R, T>>>
    splitIntoBeziers() const
    {
        // for(int dir = N-1; dir >= 0; --dir)
        for(int dir = 0; dir < N; ++dir)
        {
            const auto &unique_knots = this->getKnots(dir).getUnique();
            if (unique_knots.size() == 2)
                continue;

            const auto k = unique_knots[1];
            const auto new_splines = this->splitAtKnot(dir, k);

            auto bzrs = new_splines[0]->splitIntoBeziers();

            const auto bzrs1 = new_splines[1]->splitIntoBeziers();
            bzrs.insert(bzrs.end(), bzrs1.cbegin(), bzrs1.cend());

            return bzrs;
        }

        return {this->transformToBezier()};
    }


    /**
     * @brief Evaluates the Bspline function at a given point.
     * 
     * @tparam T_ Type of the variables.
     * @param _x Evaluation point.
     * @return Value of the function at @p _x.
     */
    template<typename T_>
    Val<T_> operator() (const uvector<T_, N>& _x) const
    {
        using CoefsTypeT_ = typename BSplineTP<N, R, T_>::CoefsType;

        const auto order = this->getOrderDir();
        const auto n_pts_dir = this->getNumPtsDir();

        std::array<std::vector<T_>, N> basis;
        uvector<int, N> basis_ind;
        for(int dir = 0; dir < N; ++dir)
        {
            const auto &bspline = this->bsplines_1D[dir];
            basis_ind(dir) = bspline->getFirstBasisIndex(_x(dir));

            basis[dir].resize(order(dir));
            bspline->evaluateBasis(_x(dir), basis[dir].data());
        }

        Val<T_> val(0.0);
        if constexpr (N == 1)
        {
            const auto *coef = coefs.data() + basis_ind(0);
            val = std::inner_product(basis[0].cbegin(), basis[0].cend(), coef, T_(0.0));
        }
        else if constexpr (N == 2)
        {
            const auto n1 = n_pts_dir(1);
            for(int i = 0; i < order(0); ++i)
            {
                const auto *coef = coefs.data() + (basis_ind(0) + i) * n1 + basis_ind(1);
                const auto v0 = std::inner_product(basis[1].cbegin(), basis[1].cend(), coef, CoefsTypeT_(0.0));

                val += basis[0][i] * v0;
            }
        }
        else if constexpr (N == 3)
        {
            const auto n1 = n_pts_dir(1);
            const auto n2 = n_pts_dir(2);

            for(int i = 0; i < order(0); ++i)
            {
                for(int j = 0; j < order(1); ++j)
                {
                    const auto *coef = coefs.data() + (basis_ind(0) + i) * n1 * n2 + (basis_ind(1) + j) * n2 + basis_ind(2);

                    const auto v0 = std::inner_product(basis[2].cbegin(), basis[2].cend(), coef, CoefsTypeT_(0.0));

                    val += basis[0][i] * basis[1][j] * v0;
                }
            }
        }
        else // if constexpr (N > 3)
        {
            assert(false);
        }

        return val;
    }

    /**
     * @brief Evaluates the gradient of the Bspline function at a given point.
     * 
     * @tparam T_ Type of the variables.
     * @param _x Evaluation point.
     * @return Value of the function's gradient at @p _x.
     */
    template<typename T_>
    Grad<T_> grad(const uvector<T_, N>& _x) const
    {
        using CoefsTypeT_ = typename BSplineTP<N, R, T_>::CoefsType;

        const auto order = this->getOrderDir();
        const auto n_pts_dir = this->getNumPtsDir();

        std::array<std::vector<T_>, N> basis;
        std::array<std::vector<T_>, N> basis_der;
        uvector<int, N> basis_ind;
        for(int dir = 0; dir < N; ++dir)
        {
            const auto &bspline = this->bsplines_1D[dir];
            basis_ind(dir) = bspline->getFirstBasisIndex(_x(dir));

            if (1 < N)
            {
                basis[dir].resize(order(dir));
                bspline->evaluateBasis(_x(dir), basis[dir].data());
            }

            basis_der[dir].resize(order(dir));
            bspline->evaluateBasisDerivative(_x(dir), basis_der[dir].data(), 1);
        }

        Grad<T_> g(0.0);
        if constexpr (N == 1)
        {
            const auto *coef = coefs.data() + basis_ind(0);
            g(0) = std::inner_product(basis_der[0].cbegin(), basis_der[0].cend(), coef, CoefsTypeT_(0.0));
        }
        else if constexpr (N == 2)
        {
            const auto n1 = n_pts_dir(1);

            for(int i = 0; i < order(0); ++i)
            {
                const auto *coef = coefs.data() + (basis_ind(0) + i) * n1 + basis_ind(1);

                const auto v0 = std::inner_product(basis[1].cbegin(), basis[1].cend(), coef, CoefsTypeT_(0.0));
                const auto v1 = std::inner_product(basis_der[1].cbegin(), basis_der[1].cend(), coef, CoefsTypeT_(0.0));

                g(0) += basis_der[0][i] * v0;
                g(1) += basis[0][i] * v1;
            }
        }
        else if constexpr (N == 3)
        {
            const auto n1 = n_pts_dir(1);
            const auto n2 = n_pts_dir(2);

            for(int i = 0; i < order(0); ++i)
            {
                for(int j = 0; j < order(1); ++j)
                {
                    const auto *coef = coefs.data() + (basis_ind(0) + i) * n1 * n2 + (basis_ind(1) + j) * n2 + basis_ind(2);

                    const auto v0 = std::inner_product(basis[2].cbegin(), basis[2].cend(), coef, CoefsTypeT_(0.0));
                    const auto v1 = std::inner_product(basis_der[2].cbegin(), basis_der[2].cend(), coef, CoefsTypeT_(0.0));

                    g(0) += basis_der[0][i] * basis[1][j] * v0;
                    g(1) += basis[0][i] * basis_der[1][j] * v0;
                    g(2) += basis[0][i] * basis[1][j] * v1;
                }
            }
        }
        else // if constexpr (3 < N)
        {
            assert(false);
        }

        return g;
    }
    /**
     * @brief Evaluates the Hessian of the Bspline function at a given point.
     * 
     * @tparam T_ Type of the variables.
     * @param _x Evaluation point.
     * @return Value of the function's Hessian at @p _x.
     */
    template<typename T_>
    Hess<T_> hessian(const uvector<T_, N>& _x) const
    {
        using CoefsTypeT_ = typename BSplineTP<N, R, T_>::CoefsType;

        const auto order = this->getOrderDir();
        const auto n_pts_dir = this->getNumPtsDir();

        std::array<std::vector<T_>, N> basis;
        std::array<std::vector<T_>, N> basis_der;
        std::array<std::vector<T_>, N> basis_der2;
        uvector<int, N> basis_ind;
        for(int dir = 0; dir < N; ++dir)
        {
            const auto &bspline = this->bsplines_1D[dir];
            basis_ind(dir) = bspline->getFirstBasisIndex(_x(dir));

            if (1 < N)
            {
                basis[dir].resize(order(dir));
                bspline->evaluateBasis(_x(dir), basis[dir].data());

                basis_der[dir].resize(order(dir));
                bspline->evaluateBasisDerivative(_x(dir), basis_der[dir].data(), 1);
            }

            basis_der2[dir].resize(order(dir));
            bspline->evaluateBasisDerivative(_x(dir), basis_der2[dir].data(), 2);
        }

        Hess<T_> h(0.0);
        if constexpr (N == 1)
        {
            const auto *coef = coefs.data() + basis_ind(0);
            h(0) = std::inner_product(basis_der2[0].cbegin(), basis_der2[0].cend(), coef, CoefsTypeT_(0.0));
        }
        else if constexpr (N == 2)
        {
            const auto n1 = n_pts_dir(1);

            for(int i = 0; i < order(0); ++i)
            {
                const auto *coef = coefs.data() + (basis_ind(0) + i) * n1 + basis_ind(1);

                const auto v0 = std::inner_product(basis[1].cbegin(), basis[1].cend(), coef, CoefsTypeT_(0.0));
                const auto v1 = std::inner_product(basis_der[1].cbegin(), basis_der[1].cend(), coef, CoefsTypeT_(0.0));
                const auto v2 = std::inner_product(basis_der2[1].cbegin(), basis_der2[1].cend(), coef, CoefsTypeT_(0.0));

                h(0) += basis_der2[0][i] * v0;
                h(1) += basis_der[0][i] * v1;
                h(2) += basis[0][i] * v2;
            }
        }
        else if constexpr (N == 3)
        {
            const auto n1 = n_pts_dir(1);
            const auto n2 = n_pts_dir(2);

            for(int i = 0; i < order(0); ++i)
            {
                for(int j = 0; j < order(1); ++j)
                {
                    const auto *coef = coefs.data() + (basis_ind(0) + i) * n1 * n2 + (basis_ind(1) + j) * n2 + basis_ind(2);

                    const auto v0 = std::inner_product(basis[2].cbegin(), basis[2].cend(), coef, CoefsTypeT_(0.0));
                    const auto v1 = std::inner_product(basis_der[2].cbegin(), basis_der[2].cend(), coef, CoefsTypeT_(0.0));
                    const auto v2 = std::inner_product(basis_der2[2].cbegin(), basis_der2[2].cend(), coef, CoefsTypeT_(0.0));

                    h(0) += basis_der2[0][i] * basis[1][j] * v0;
                    h(1) += basis_der[0][i] * basis_der[1][j] * v0;
                    h(2) += basis_der[0][i] * basis[1][j] * v1;
                    h(3) += basis[0][i] * basis_der2[1][j] * v0;
                    h(4) += basis[0][i] * basis_der[1][j] * v1;
                    h(5) += basis[0][i] * basis[1][j] * v2;
                }
            }
        }
        else // if constexpr (3 < N)
        {
            assert(false);
        }

        return h;
    }

    /**
     * @brief Computes lower and upper bounds of the spline's image
     * and study if they change sign.
     * 
     * The bounds are computed by studying the convex hull of the control points.
     * 
     * @return If the image does not change sign, returns either +1 or -1, depending if the image
     * is fully positive or fully negative. Otherwise, if it changes sign (or it is uncertain), returns 0.
     */
    template<int R_ = R,
             std::enable_if_t<R_ == 1 && R_ == R, bool> = true>
    int studyChangeOfSign() const
    {
        constexpr real tolerance = std::numeric_limits<T>::epsilon() * 10.0;

        const bool has_negative = std::any_of(coefs.cbegin(), coefs.cend(),
        [tolerance](const auto &val) {return val < -tolerance;});

        if (!has_negative)
        {
            return 1; // All values are positive.
        }
        else
        {
            const bool has_positive = std::any_of(coefs.cbegin(), coefs.cend(),
            [tolerance](const auto &val) {return val > tolerance;});

            return has_positive ? 0 /* Change of sign */ : -1 /* All negative */;
        }
    }

    /**
     * @brief Studies the monotonicity of a scalar BSpline along the N parametric directions.
     * 
     * It actually studies an conservative bound of the monotonicity by studying
     * for all the control point lines along each parametric direction.
     * 
     * @return Whether each parametric direction is monotone (true) or not (false).
     */
    template<typename TT = T, std::enable_if_t<std::is_floating_point_v<TT>, bool> = true>
    uvector<bool, N> studyMonotonicity() const
    {
        constexpr real tolerance = std::numeric_limits<T>::epsilon() * 10.0;

        const auto n_pts = this->getNumPtsDir();

        uvector<bool, N> monotone(true);
        for(int dir = 0; dir < N; ++dir)
        {
            int step = 1;
            for(int dir2 = N-1; dir2 >= 0; --dir2)
                step *= n_pts(dir2);
            const auto n = n_pts(dir);

            const auto i1 = set_component(n_pts, dir, 1);
            for (MultiLoop<N> i(0, i1); ~i; ++i)
            {
                const auto *c = coefs.data() + util::toTensorIndex<N>(n_pts, i());

                // First we detect the tendency (positive or negative).
                T v0 = *c;
                bool increasing = true;

                int j;
                for(j = 1; j < n; ++j)
                {
                    c += step;
                    const auto diff = *c - v0;
                    v0 = *c;

                    if (diff > tolerance)
                        break;
                    else if (diff < -tolerance)
                        increasing = false;
                        break;
                }

                // Then we study if the tendency persists.
                for(++j; j < n; ++j)
                {
                    c += step;
                    const auto diff = *c - v0;
                    v0 = *c;

                    if ((!increasing && diff >  tolerance) ||
                        ( increasing && diff < -tolerance))
                    {
                        monotone(dir) = false;
                        break;
                    }
                }

                if (!monotone(dir))
                    break;
            }
        }
        return monotone;
    }

    /**
     * @brief Computes the L2 projection of the function @p _func into the current spline space.
     * @tparam F Type of the function to project.
     * @param _func Function to project.
     * @param _n_sample_pts_per_elem Number of Gauss points per direction to be used in the projection.
     * 
     * @note Currently, only the case of scalar functions is suported.
     */
    template<typename F, typename Taux = T>
    void projectL2(const F &_func,
                   const uvector<int, N> &_n_sample_pts_per_elem,
                   std::enable_if_t<std::is_same_v<Taux, real>, void *> = nullptr)
    {
        std::array<std::shared_ptr<const BSplineEvaluation>, N> bspline_evals;

        bspline_evals[0] = bsplines_1D[0]->evaluate(_n_sample_pts_per_elem(0));
        for(int dir = 1; dir < N; ++dir)
        {
            if (bsplines_1D[dir] == bsplines_1D[0] && _n_sample_pts_per_elem(dir) == _n_sample_pts_per_elem(0))
                bspline_evals[dir] = bspline_evals[0];
            else
                bspline_evals[dir] = bsplines_1D[dir]->evaluate(_n_sample_pts_per_elem(dir));
        }

        compute_f_v(_func, _n_sample_pts_per_elem, bspline_evals);

        computeCoefsL2(bspline_evals);
    }

private:
    /**
     * @brief Computes the integral @p _func tested against the function in the spline space.
     * 
     * This is required for computing the L2 projection of @p _func.
     * 
     * @tparam F Type of the function to project.
     * @param _func Function to project.
     * @param _n_pts Number of Gauss points per direction to be used in the projection.
     * @param _bspline_evals 1D BSpline evaluations along each parametric direction.
     * 
     * @note Currently, only the case of scalar functions is suported.
     */
    template<typename F, typename Taux = T>
    void compute_f_v(const F &_func,
                     const uvector<int, N> &_n_pts,
                     const std::array<std::shared_ptr<const BSplineEvaluation>, N> &_bspline_evals,
                     std::enable_if_t<std::is_same_v<Taux, real>, void *> = nullptr)
    {
        static_assert(N == 2 || N == 3, "Invalid dimension.");

        const auto n_funcs_dir = this->getNumPtsDir();
        const auto order = this->getOrderDir();

        uvector<int, N> n_elems;
        for(int dir = 0; dir < N; ++dir)
            n_elems(dir) = _bspline_evals[dir]->n_elems;

        auto &f_v = this->coefs;
        f_v.assign(prod(n_funcs_dir), 0.0);


        uvector<int, N> accum_ids(1);
        for(int dir = N-2; dir >= 0; --dir)
            accum_ids(dir) = accum_ids(dir+1) * n_funcs_dir(dir+1);


        #pragma omp parallel for
        for(int el_id = 0; el_id < prod(n_elems); ++el_id)
        {
            const auto el_tid = util::toTensorIndex<N>(n_elems, el_id);

            std::vector<int> ids(prod(order), 0);
            auto ids_it = ids.begin();
            for (MultiLoop<N> fc_tid(0, order); ~fc_tid; ++fc_tid, ++ids_it)
            {
                for(int dir = 0; dir < N; ++dir)
                    *ids_it += (_bspline_evals[dir]->first_func_per_elem[el_tid(dir)] + fc_tid(dir)) * accum_ids(dir);
            }

            std::vector<real> values(prod(order), 0.0);
            uvector<real, N> x;
            for (MultiLoop<N> pt_tid(0, _n_pts); ~pt_tid; ++pt_tid)
            {
                for(int dir = 0; dir < N; ++dir)
                    x(dir) = _bspline_evals[dir]->param_coords[el_tid(dir) * _n_pts(dir) + pt_tid(dir)];

                const auto f = _func(x);

                auto values_it = values.begin();
                for (MultiLoop<N> fc_tid(0, order); ~fc_tid; ++fc_tid, ++values_it)
                {
                    real val = f;
                    for(int dir = 0; dir < N; ++dir)
                        val *= _bspline_evals[dir]->basis_values_w[el_tid(dir) * _n_pts(dir) * order(dir) + pt_tid(dir) * order(dir) + fc_tid(dir)];

                    *values_it += val;
                }
            }

            #pragma omp critical
            {
                for(int i = 0; i < prod(order); ++i)
                    f_v[ids[i]] += values[i];
            }
        }

    }

    /**
     * @brief Computes the coefficients of the L2 projection of @p _func.
     * 
     * @param _bspline_evals 1D BSpline evaluations along each parametric direction.
     * 
     * @note Currently, only the case of scalar functions is suported.
     */
    template<typename Taux = T>
    void computeCoefsL2(const std::array<std::shared_ptr<const BSplineEvaluation>, N> &_bspline_evals,
                          std::enable_if_t<std::is_same_v<Taux, real>, void *> = nullptr)
    {
        static_assert(1 <= N &&  N <= 3, "Invalid dimension.");

        const auto M0 = _bspline_evals[0]->computeMass();
        SymBandMatrixCholesky invM0(*M0);

        if constexpr (N == 1)
        {
            invM0.solve(this->coefs);
        }
        else if constexpr (N == 2)
        {
            if (_bspline_evals[0] == _bspline_evals[1])
            {
                SymBandMatrixCholesky::solveKronecker(invM0, invM0, this->coefs);

            }
            else
            {
                const auto M1 = _bspline_evals[1]->computeMass();
                SymBandMatrixCholesky invM1(*M1);
                SymBandMatrixCholesky::solveKronecker(invM0, invM1, this->coefs);
            }
        }
        else if constexpr (N == 3)
        {
            if (_bspline_evals[0] == _bspline_evals[1] && _bspline_evals[0] == _bspline_evals[2])
            {
                SymBandMatrixCholesky::solveKronecker(invM0, invM0, invM0, this->coefs);
            }
            else
            {
                const auto M1 = _bspline_evals[1]->computeMass();
                const auto M2 = _bspline_evals[2]->computeMass();
                SymBandMatrixCholesky invM1(*M1);
                SymBandMatrixCholesky invM2(*M2);

                SymBandMatrixCholesky::solveKronecker(invM0, invM1, invM2, this->coefs);
            }
        }
        else // if constexpr (N > 3)
        {
            assert(false);
        }
    }

};

} // namespace algoim::bspline

#endif // ALGOIM_BSPLINE_H
