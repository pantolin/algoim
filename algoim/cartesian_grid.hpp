#ifndef ALGOIM_CARTESIAN_GRID_H
#define ALGOIM_CARTESIAN_GRID_H

#include "hyperrectangle.hpp"
#include "real.hpp"
#include "uvector.hpp"
#include "multiloop.hpp"
#include "utility.hpp"

#include <memory>
#include <cassert>

namespace algoim
{

/**
 * @brief Cartesian tensor-product grid.
 * 
 * @tparam N Parametric dimension.
 */
template<int N>
struct Grid
{
    /**
     * @brief Constructor.
     * Creates a grid in the hypercube [0,1]^N
     * with @p _n_elems_dir per direction.
     * 
     * @param _n_elems_dir Number of elements per direction.
     */
    Grid(std::array<std::vector<real>, N> _data) : data(_data)
    {
        #ifndef NDEBUG
        const auto n_elems_dir = this->getNumElemsDir();
        for(int dir = 0; dir < N; ++dir)
        {
            for(int i = 0; i < n_elems_dir(dir); ++i)
            {
                assert(data[dir][i] < data[dir][i+1]);
            }
        }
        #endif // NDEBUG
    }

    /**
     * @brief Constructor.
     * Creates a grid in the hypercube [0,1]^N
     * with @p _n_elems_dir per direction.
     * 
     * @param _n_elems_dir Number of elements per direction.
     */
    Grid(const uvector<real, N> &_xmin,
         const uvector<real, N> &_xmax,
         const uvector<int, N> &_n_elems_dir)
    {
        for(int dir = 0; dir < N; ++dir)
        {
            assert(0 < _n_elems_dir(dir));
            const real dx = (_xmax(dir) - _xmin(dir)) / static_cast<real>(_n_elems_dir(dir));

            data[dir].reserve(_n_elems_dir(dir) + 1);
            for(int i = 0; i <= _n_elems_dir(dir); ++i)
                data[dir].push_back(_xmin(dir) + i * dx);
        }
    }

    /**
     * @brief Constructor.
     * Creates a grid in the hypercube [0,1]^N
     * with @p _n_elems_dir per direction.
     * 
     * @param _n_elems_dir Number of elements per direction.
     */
    Grid(const uvector<int, N> &_n_elems_dir)
      :
      Grid(uvector<real,N>(0.0), uvector<real,N>(1.0), _n_elems_dir)
    {
    }

    /// Grid coordinates along each parametric direction.
    std::array<std::vector<real>, N> data;

    /**
     * @brief Gets the number of elements (spans) per direction.
     * 
     * @return Number of elements per direction.
     */
    uvector<int, N> getNumElemsDir() const
    {
        uvector<int, N> n_elems_dir;
        for(int dir = 0; dir < N; ++dir)
            n_elems_dir(dir) = static_cast<int>(data[dir].size()) - 1;
        return n_elems_dir;
    }

    /**
     * @brief Creates the bounding box of the given element.
     * 
     * @return Bounding box of the element.
     */
    HyperRectangle<real, N> getDomain(const int _elem_id) const
    {
        const auto elem_tid = util::toTensorIndex(this->getNumElemsDir(), _elem_id);
        HyperRectangle<real, N> domain(0, 1);
        for(int dir = 0; dir < N; ++dir)
        {
            domain.min(dir) = this->data[dir][elem_tid(dir)];
            domain.max(dir) = this->data[dir][elem_tid(dir) + 1];
        }

        return domain;
    }

};

/**
 * @brief Subgrid of a Cartesian grid.
 * It is a subset of the elements of a given grid.
 * 
 * @tparam N Parametric dimension.
 */
template<int N>
struct SubGrid
{
    /**
     * @brief Constructor.
     * 
     * @param _grid Parent grid.
     * @param _indices_start Start indices of the coordinates of the parent grid.
     * @param _indices_end End indices of the coordinates of the parent grid.
     */
    SubGrid(const Grid<N> &_grid,
            const uvector<int, N> &_indices_start,
            const uvector<int, N> &_indices_end)
    : grid(_grid), indices_start(_indices_start), indices_end(_indices_end)
    {
        assert(0 < this->getNumElems());
    }

    /**
     * @brief Constructor.
     * Creates a subgrid containing the full grid.
     * 
     * @param _grid Parent grid.
     */
    SubGrid(const Grid<N> &_grid)
    : SubGrid(_grid, 0, _grid.getNumElemsDir())
    {}

private:
    /// Parent grid.
    const Grid<N> &grid;
    /// Start indices of the coordinates of teh parent grid.
    uvector<int, N> indices_start;
    /// End indices of the coordinates of teh parent grid.
    uvector<int, N> indices_end;

public:

    /**
     * @brief Gets the number of elements (spans) per direction
     * of the subgrid. 
     * 
     * The ordering is such that dimension N-1 is inner-most, i.e.,
     * iterates the fastest, while dimension 0 is outer-most and iterates the slowest.
     * 
     * @return Number of elements per direction.
     */
    uvector<int, N> getNumElemsDir() const
    {
        uvector<int, N> n_elems_dir;
        for(int dir = 0; dir < N; ++dir)
            n_elems_dir(dir) = indices_end(dir) - indices_start(dir);
        return n_elems_dir;
    }

    /**
     * @brief Gets the number of elements of the subgrid.
     * 
     * @return Number of elements.
     */
    int getNumElems() const
    {
        return prod(this->getNumElemsDir());
    }

    /**
     * @brief Checks if the subgrid has only one element.
     * 
     * @return True if it has only one element, false otherwise.
     */
    bool uniqueElement() const
    {
        return this->getNumElems() == 1;
    }

    /**
     * @brief Splits the current subgrid along the direction
     * with a largest number of elements.
     * 
     * @return Two generated subgrid wrapped in shared pointers.
     */
    std::array<std::shared_ptr<const SubGrid<N>>, 2> split() const
    {
        assert(!this->uniqueElement());

        const auto n_elems_dir = this->getNumElemsDir();
        const auto split_dir = argmax(n_elems_dir);

        uvector<int, N> new_indices_start = indices_start;
        uvector<int, N> new_indices_end = indices_end;

        const int split_id = (indices_start(split_dir) + indices_end(split_dir)) / 2;

        std::array<std::shared_ptr<const SubGrid<N>>, 2> subgrids;

        new_indices_end(split_dir) = split_id;
        subgrids[0] = std::make_shared<SubGrid>(grid, indices_start, new_indices_end);

        new_indices_start(split_dir) = split_id;
        subgrids[1] = std::make_shared<SubGrid>(grid, new_indices_start, indices_end);

        return subgrids;
    }

    /**
     * @brief Creates the bounding box of the subgrid's domain.
     * 
     * @return Bounding box of the subgrid.
     */
    HyperRectangle<real, N> getDomain() const
    {
        HyperRectangle<real, N> domain(0, 1);
        for(int dir = 0; dir < N; ++dir)
        {
            domain.min(dir) = grid.data[dir][indices_start(dir)];
            domain.max(dir) = grid.data[dir][indices_end(dir)];
        }

        return domain;
    }

    /**
     * @brief Gets the indices of the subgrid respect to the
     * numeration of the parent grid.
     * 
     * @return List of element indices.
     */
    std::vector<int> getElementIndices() const
    {
        const auto n_elems_dir = grid.getNumElemsDir();

        std::vector<int> element_indices;
        element_indices.reserve(this->getNumElems());

        for(MultiLoop<N> i(indices_start, indices_end); ~i; ++i)
        {
            const auto elem_id = util::toFlatIndex(n_elems_dir, i());
            element_indices.push_back(elem_id);
        }

        return element_indices;
    }
};


} // namespace algoim::general

#endif // ALGOIM_CARTESIAN_GRID_H
