#ifndef ALGOIM_QUADRATURE_ELEMENT_H
#define ALGOIM_QUADRATURE_ELEMENT_H

/* Data structures for storing element quadratures. */

#include "real.hpp"
#include "uvector.hpp"
#include "hyperrectangle.hpp"
#include "quadrature_general.hpp"

#include <map>
#include <memory>
#include <vector>

namespace algoim::quad
{
    namespace detail
    {
        /**
         * @brief Computes the normal direction to an isoparametric boundary.
         * 
         * @tparam N Dimension of the hypercube.
         * @param _iso_bound_id Id of the face (in [0, 2*N[) in lexicograhical convention.
         * @return Computed area.
         */
        template<int N>
        uvector<real, N> computeIsoBdryNormal(const int _iso_bound_id)
        {
            const int const_dir = _iso_bound_id / N;
            const int side = _iso_bound_id % N;

            uvector<real, N> normal(0.0);
            normal(const_dir) = side == 0 ? -1.0 : 1.0;
            return normal;
        }

    } // namespace detail

    /**
     * @brief Data structure for storing quadratures (point coordinates and weights).
     * 
     * @tparam N Parametric dimension of the quadrature.
     */
    template<int N>
    struct Quadrature
    {
        /// Vector of points.
        std::vector<uvector<real, N>> points;
        /// Vector of weights.
        std::vector<real> weights;

        /**
         * @brief Clears (resets) the data structure.
         */
        virtual void clear()
        {
            points.clear();
            weights.clear();
        }

        /**
         * @brief Updates the quadrature by appending new nodes (points and weights).
         * 
         * @tparam Node Node type (including point and weight).
         * @param _nodes Vector of nodes for the quadrature.
         */
        template<typename Node>
        void fill(const std::vector<Node> &_nodes)
        {
            const auto n = _nodes.size();
            this->points.resize(n);
            this->weights.resize(n);

            for(std::size_t i = 0; i < n; ++i)
            {
                this->points[i] = _nodes[i].x;
                this->weights[i] = _nodes[i].w;
            }
        }
    };


    /**
     * @brief Data structure for storing boundary quadratures (point coordinates, weights, and normals).
     * 
     * @tparam N Parametric dimension of the quadrature.
     */
    template<int N>
    struct QuadratureBoundary : public Quadrature<N>
    {
        /// Vector of unit normals at quadrature points.
        std::vector<uvector<real,N>> normals;

        /**
         * @brief Clears (resets) the data structure.
         */
        void clear() override
        {
            Quadrature<N>::clear();
            normals.clear();
        }

        /**
         * @brief Updates the quadrature by appending new nodes (points, weights, and normals).
         * 
         * @tparam Node Node type (including point and weight).
         * @tparam F Type of implicit function defining the domain.
         * @param _nodes Vector of nodes for the quadrature.
         * @param _phi Implicit function used for computing the normals (along the gradients of @p _phi).
         */
        template<typename Node, typename F>
        void fill(const std::vector<Node> &_nodes, const F &_phi)
        {
            Quadrature<N>::fill(_nodes);

            const auto n = this->points.size();
            normals.resize(n);

            for(std::size_t i = 0; i < n; ++i)
            {
                const auto &x = this->points[i];
                normals[i] = _phi.grad(x);
                normals[i] /= norm(normals[i]);
            }

        }
    };

    /**
     * @brief Data structure for storing isoparametri boundary quadratures
     * (point coordinates, weights, and normals).
     * 
     * @tparam N Parametric dimension of the quadrature.
     */
    template<int N>
    struct QuadratureIsoBoundary : public QuadratureBoundary<N>
    {
        /// Flag indicating if the boundary is cut (intersected by the function levelset) or not.
        bool cut{false};
        /// Flag indicating if, for a non cut boundary, the element is inside the domain
        /// (the domain function is negative), or outside.
        bool interior{true};

        /**
         * Clears the data structure.
         */
        void clear() final
        {
            QuadratureBoundary<N>::clear();
            cut = false;
            interior = true;
        }

        /**
         * @brief Updates the quadrature by appending new nodes (points, weights, and normals).
         * 
         * @tparam Node Node type (including point and weight).
         * @param _nodes Vector of nodes for the quadrature.
         * @param _iso_bdry_id Id of the iso boundary (in range [0, 2*N[) in lexicograhical convention.
         * 
         * @note The normal direction is deduced from the boundary @p _iso_bdry_id.
         */
        template<typename Node>
        void fillIso(const std::vector<Node> &_nodes, const int _iso_bdry_id)
        {
            Quadrature<N>::fill(_nodes);
            this->fillNormaliso(static_cast<int>(_nodes.size()), _iso_bdry_id);
        }

        /**
         * @brief Updates the quadrature normals.
         * 
         * @param _n_points Number of points of the quadrature.
         * @param _iso_bdry_id Id of the iso boundary (in range [0, 2*N[) in lexicograhical convention.
         * 
         * @note The normal direction is deduced from the boundary @p _iso_bdry_id.
         */
        void fillNormaliso(const int _n_points, const int _iso_bdry_id)
        {
            const auto normal = detail::computeIsoBdryNormal<N>(_iso_bdry_id);
            this->normals.assign(_n_points, normal);
        }

    };

    /**
     * @brief Data structure for holding the quadrature of an element, including its boundary.
     * It also provides information regarding if the element is cut or not, interior or not.
     * 
     * @tparam N Parametric dimension of the quadrature.
     */
    template<int N>
    struct ElementQuadrature
    {
        /**
         * Clears the data structure.
         */
        void clear()
        {
            cut = false;
            interior = true;
            quad.clear();
            bdry_quads.clear();
        }

        /// Flag indicating if the element is cut (intersected by the function levelset) or not.
        bool cut{false};
        /// Flag indicating if, for a non cut element, the element is inside the domain
        /// (the domain function is negative), or outside.
        bool interior{true};
        /// Quadrature for the element's interior (only for cut elements).
        Quadrature<N> quad;
        /// Quadratures for the boundary (levelset of the implicit function) of the element (only for cut elements).
        /// A name (string) is associated to every quadrature.
        std::map<std::string, QuadratureBoundary<N>> bdry_quads;
        /// Quadrature for isoparametric boundaries of the element (only for cut elements).
        /// The indices in the array follow the lexicographical convention.
        std::array<QuadratureIsoBoundary<N>, N*2> iso_bdry_quad;
    };



} // namespace algoim::quad

#endif // ALGOIM_QUADRATURE_ELEMENT_H
