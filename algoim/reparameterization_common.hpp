#ifndef ALGOIM_REPARAMETERIZATION_COMMON_H
#define ALGOIM_REPARAMETERIZATION_COMMON_H

/* Common tools for creating reparameterizations. */

#include "hyperrectangle.hpp"
#include "polynomial_tp.hpp"
#include "real.hpp"
#include "uvector.hpp"
#include "lagrange.hpp"
#include "utility.hpp"

#include <limits>
#include <map>


namespace algoim::detail
{
    /// Reparameterization element type.
    template<int M, int N>
    using ReparamElemType = lagrange::LagrangeTP<M,N>;

    /**
     * @brief Struct for storing reparameterization elements.
     * 
     * @tparam M Parametric dimension of the elements.
     * @tparam N Range of the elements.
     */
    template<int M, int N>
    struct ReparamElems
    {
        /// Element type.
        using ElemType = ReparamElemType<M, N>;
        /// Shared pointer element type.
        using ElemTypePtr = std::shared_ptr<ElemType>;

        /**
         * @brief Default constructor.
         */
        ReparamElems() = default;

        /**
         * @brief Constructor.
         * 
         * @param _p Reparameterization order (number of points per direction).
         */
        ReparamElems(const int _p) : p(_p)
        {
            assert(1 < p);
        }


        /// Reparameterization order (number of points per direction).
        int p;
        /// Vector of reparameterization elements.
        std::vector<ElemTypePtr> reparam_elems;
        /// Map of reparameterization elements (element tensor index to element). Each element
        /// in this container also belongs to reparam_elems (the contrary is not necessary true).
        std::map<uvector<int,N>, ElemTypePtr, util::UvectorCompare<int, N>> reparam_elems_map;


        /**
         * @brief This method sets the given point @p _x, with index @p _pt_tid,
         * to the element designed by _elem_tid.
         * 
         * @param _x Point to be set.
         * @param _elem_tid Tensor index of the element in which the point is set.
         * @param _pt_tid Tensor index of the reparameterization point referred to the element.
         */
        void process(uvector<real,N> _x, uvector<int,N> _elem_tid, uvector<int,M> _pt_tid)
        {

            ElemTypePtr elem{nullptr};
            if (all(_pt_tid == 0))
            {
                const uvector<int,M> order(this->p);
                elem = std::make_shared<ElemType>(order);
                this->reparam_elems_map.emplace(_elem_tid, elem);
                this->reparam_elems.push_back(elem);
            }
            else
            {
                elem = this->reparam_elems_map.at(_elem_tid);
            }

            const auto flat_id =elem->toFlatIndex(_pt_tid);
            elem->coefs[flat_id] = _x;
        }

        /**
         * @brief Clears the map of reparameterization elements.
         */
        void clearReparamElemsMap()
        {
            this->reparam_elems_map.clear();
        }

        /**
         * @brief Computes the reference intervals.
         * @note This is a dummy method to not break the dimension recursivity.
         */
        void computeReferenceIntervals(uvector<real,N> _x, uvector<int,N> _elem_tid) const
        {
            // Do nothing.
        }

        /**
         * @brief Clears recursively the reference intervals.
         * @note This is a dummy method to not break the dimension recursivity.
         */
        void clearReferenceIntervals() const
        {
            // Do nothing.
        }
    };

    /**
     * @brief Struct for storing and managing computed roots and intervals
     * of an implicit function.
     * @tparam N Dimension of the point at which the intervals are computed.
     */
    template<int N>
    struct RootsIntervals
    {
        /// List of roots.
        std::vector<real> roots;
        /// Point at which roots are computed.
        uvector<real,N> point;
        /// Restrictions to which root correspond to (there is a one to correspondence).
        std::vector<int> func_ids;
        /// Flags indicating if the intervals defined by two consecutive roots are active.
        std::vector<bool> active_intervals;

        /**
         * @brief Adds a new root.
         * 
         * @param _root New root to be added.
         * @param _func_id If of the restriction to which the @p _root belongs to.
         * 
         * @note The roots are neither sorted nor adjusted (for degeneracies) after
         * appending the new root.
         */
        void addRoot(const real _root, const int _func_id)
        {
            this->roots.push_back(_root);
            this->func_ids.push_back(_func_id);
        }

        /**
         * @brief Checks whether the container is empty
         * 
         * @return True if empty, i.e. there are no roots, false otherwise.
         */
        bool empty() const
        {
            return this->roots.empty();
        }

        /**
         * @brief Gets the number of roots in the container.
         * 
         * @return Number of roots.
         */
        int getNumRoots() const
        {
            return static_cast<int>(this->roots.size());
        }

        /**
         * @brief Sorts (in increasing order) the roots in the container
         * and the according restriction indices func_ids.
         */
        void sortRoots()
        {
            const auto n = this->getNumRoots();

            // Zip.
            std::vector<std::pair<real,int>> roots_ids;
            roots_ids.reserve(n);
            for(int i = 0; i < n; ++i)
                roots_ids.emplace_back(this->roots[i], this->func_ids[i]);

            // Sort.
            std::sort(roots_ids.begin(), roots_ids.end(),
                [this](auto &_a, auto &_b) -> bool
                {
                    return _a.first < _b.first;
                });

            // Unzip.
            for(int i = 0; i < n; ++i)
            {
                this->roots[i] = roots_ids[i].first;
                this->func_ids[i] = roots_ids[i].second;
            }
        }

        /**
         * @brief Adjust the container roots by sorting them
         * and forcing near roots (up to a @p _tolerance) to be coincident.
         * 
         * @param _tolerance Tolerance to be used in the comparisons between
         *        roots.
         * @param _x0 Start of the interval to which the roots belong to.
         * @param _x1 End of the interval to which the roots belong to.
         */
        void adjustRoots(const real _tolerance, const real _x0, const real _x1)
        {
            if (roots.empty())
                return;

            this->sortRoots();

            for (std::size_t i = 0; i < roots.size(); ++i)
            {
                if (std::abs(this->roots[i] - _x0) < _tolerance)
                    this->roots[i] = _x0;
                else if (std::abs(this->roots[i] - _x1) < _tolerance)
                    this->roots[i] = _x1;
                else if (0 < i && std::fabs(roots[i] - roots[i-1]) < _tolerance)
                    this->roots[i] = this->roots[i-1];
            } // i

            // Enforcing root=_x0, func_id=-1, and root=_x1, func_id=-1, to be the first
            // and last, respectively.

            for (std::size_t i = 1; i < roots.size(); ++i)
            {
                if (this->func_ids[i] == -1 && std::abs(this->roots[i] - _x0) < _tolerance)
                {
                    this->roots.erase(std::next(this->roots.begin(), i));
                    this->func_ids.erase(std::next(this->func_ids.begin(), i));
                    this->roots.insert(this->roots.begin(), _x0);
                    this->func_ids.insert(this->func_ids.begin(), -1);
                    break;
                }
            } // i

            for (std::size_t i = 0; i < (static_cast<int>(roots.size()) - 1); ++i)
            {
                if (this->func_ids[i] == -1 && std::abs(this->roots[i] - _x1) < _tolerance)
                {
                    this->roots.erase(std::next(this->roots.begin(), i));
                    this->func_ids.erase(std::next(this->func_ids.begin(), i));
                    this->roots.push_back(_x1);
                    this->func_ids.push_back(-1);
                }
            } // i
        }
    };

    /**
     * @brief Extracts the edges of Lagrange hypercube of the given @p _order.
     * 
     * @tparam N Dimension of the hypercube.
     * @param _order Order of the create edges.
     * @return Created edges.
     */
    template<int N>
    std::vector<std::shared_ptr<lagrange::LagrangeTP<1, N>>>
    createHypercubeWirebasket(const int _order)
    {
        static_assert(N == 2 || N == 3, "Not implemented.");
        assert(1 < _order);

        const HyperRectangle<real, N> unit_domain(0, 1);
        const auto unit_cube = std::make_shared<lagrange::LagrangeTP<N, N>>(unit_domain, _order);

        const auto n_edges = N == 2 ? 4 : 12;

        std::vector<std::shared_ptr<lagrange::LagrangeTP<1, N>>> edges(n_edges);
        for(int i = 0; i < n_edges; ++i)
            edges[i] = unit_cube->extractEdge(i);
        return edges;
    }

    /**
     * @brief Checks if a reparameterization cell's edge belongs to a subentity of a
     * hypercube domain.
     * 
     * @tparam M Parametric domain of the reparameterization cell.
     * @tparam N Physical domain of the reparameterization cell.
     * @tparam Q Parametric dimension of the subentity.
     * @param _rep Reparameterization cell whose edge is considered.
     * @param _edge_id Edge id of the reparameterization cell to consider.
     * @param _domain Hypercube domain being considered. 
     * @return True if the edge sits on the subentity of dimension @p Q of @p _domain,
     * false otherwise.
     */
    template<int M, int N, int Q>
    static bool checkEdgeInDomainSubEntity(const ReparamElemType<M,N> &_rep,
                                           const int _edge_id,
                                           const HyperRectangle<real, N> &_domain)
    {
        const auto get_bounds = [&_domain](const auto &_pt)
        {
            constexpr real tol = std::numeric_limits<real>::epsilon() * 10.0;

            uvector<bool, N> bounds;
            for(int dir = 0; dir < N; ++dir)
            {
                bounds(dir) = std::fabs(_pt(dir) - _domain.min(dir)) < tol || 
                              std::fabs(_pt(dir) - _domain.max(dir)) < tol;
            }

            return bounds;
        };

        PolynomialTPEdgeIt<M> edge_it(_rep.order, _edge_id);
        const auto &pt0 = _rep.coefs[edge_it.getFlatIndex()];
        const auto bounds0 = get_bounds(pt0);
        
        int count = 0;
        for(int dir = 0; dir < N; ++dir)
            if (bounds0(dir)) ++count;
        if (count < Q)
            return false;

        for(++edge_it; ~edge_it; ++edge_it)
        {
            const auto &pt = _rep.coefs[edge_it.getFlatIndex()];
            const auto bounds = get_bounds(pt);
            int count = 0;
            for(int dir = 0; dir < N; ++dir)
            {
                if (bounds0(dir) && bounds(dir))
                    ++count;
            }
            if (count < Q)
                return false;
        }

        return true;
    }

    /**
     * @brief Checks if a reparameterization cell's edge belongs to an edge of a
     * hypercube domain.
     * 
     * @tparam M Parametric domain of the reparameterization cell.
     * @tparam N Physical domain of the reparameterization cell.
     * @param _rep Reparameterization cell whose edge is considered.
     * @param _edge_id Edge id of the reparameterization cell to consider.
     * @param _domain Hypercube domain being considered. 
     * @return True if the edge sits on an edge of @p _domain, false otherwise.
     */
    template<int M, int N>
    static bool checkEdgeInDomainEdge(const ReparamElemType<M,N> &_rep,
                                      const int _edge_id,
                                      const HyperRectangle<real, N> &_domain)
    {
        return checkEdgeInDomainSubEntity<M, N, N-1>(_rep, _edge_id, _domain);
    }

    /**
     * @brief Checks if a reparameterization cell's edge belongs to a face of a
     * hypercube domain.
     * 
     * @tparam M Parametric domain of the reparameterization cell.
     * @tparam N Physical domain of the reparameterization cell.
     * @param _rep Reparameterization cell whose edge is considered.
     * @param _edge_id Edge id of the reparameterization cell to consider.
     * @param _domain Hypercube domain being considered. 
     * @return True if the edge sits on a face of @p _domain, false otherwise.
     */
    template<int M, int N>
    static bool checkEdgeInDomainFace(const ReparamElemType<M,N> &_rep,
                                      const int _edge_id,
                                      const HyperRectangle<real, N> &_domain)
    {
        return checkEdgeInDomainSubEntity<M, N, 1>(_rep, _edge_id, _domain);
    }

    /**
     * @brief Checks if a reparameterization cell's edge is degenerate (it has zero length).
     * 
     * @tparam M Parametric domain of the reparameterization cell.
     * @tparam N Physical domain of the reparameterization cell.
     * @param _rep Reparameterization cell whose edge is considered.
     * @param _edge_id Edge id of the reparameterization cell to consider.
     * @return True if the edge is degenerate, false otherwise.
     */
    template<int M, int N>
    static bool checkEdgeDegenerate(const ReparamElemType<M,N> &_rep,
                                    const int _edge_id)
    {
        constexpr real tol = std::numeric_limits<real>::epsilon() * 10.0;

        PolynomialTPEdgeIt<M> edge_it(_rep.order, _edge_id);
        const auto &pt0 = _rep.coefs[edge_it.getFlatIndex()];

        bool degenerate{true};
        for(++edge_it; ~edge_it; ++edge_it)
        {
            const auto &pt = _rep.coefs[edge_it.getFlatIndex()];
            if (tol < norm(pt0 - pt))
                return false;
        }

        return true;
    }

    /**
     * @brief Extracts the wirebasket of a given reparameterization.
     * 
     * @tparam M Parametric dimension of the reparameterization.
     * @tparam N Physical dimension of the reparameterization.
     * @tparam S True if the reparameterization corresponds to a levelset.
     * @param _reparam Reparameterization whose wirebasket is extracted.
     * @param _domain Domain (hypercube) to which the reparameterization corresponds.
     * @param _check_face Function returning true if an edge in a domain's face must
     *        be included in the wirebasket.
     * @param _check_internal Function returning true if an edge that does not correspond
     *        to neither a face nor an edge of @p _domain must be included in the wirebasket.
     * @return 
     */
    template<int M, int N, bool S = false>
    std::vector<std::shared_ptr<algoim::lagrange::LagrangeTP<1,N>>>
    extractWirebasket(
        const std::vector<std::shared_ptr<ReparamElemType<M,N>>> &_reparam,
        const HyperRectangle<real, N> &_domain,
        const std::function<bool (const uvector<real,N> &)> &_check_face,
        const std::function<bool (const uvector<real,N> &)> &_check_internal)
    {
        static_assert(2 <= M && M <= 3, "Invalid dimension.");

        std::vector<std::shared_ptr<algoim::lagrange::LagrangeTP<1,N>>> edges;
        edges.reserve(_reparam.size() * 6); // Just a conservative estimate.

        const int n_edges = M == 2 ? 4 : 12;
        for(const auto &rep : _reparam)
        {
            for(int edge_id = 0; edge_id < n_edges; ++edge_id)
            {

                if (checkEdgeDegenerate(*rep, edge_id))
                    continue;

                auto add_edge = !S && checkEdgeInDomainEdge(*rep, edge_id, _domain);

                if (!add_edge)
                {
                    if (checkEdgeInDomainFace(*rep, edge_id, _domain))
                    {
                        bool is_valid{true};
                        for(PolynomialTPEdgeIt<M> edge_it(rep->order, edge_id); ~edge_it; ++edge_it)
                        {
                            const auto &pt = rep->coefs[edge_it.getFlatIndex()];
                            is_valid = _check_face(pt);
                            if (!is_valid)
                                break;
                        }
                        add_edge = is_valid;
                    }
                }

                if (!add_edge)
                {
                    bool is_valid{true};

                    for(PolynomialTPEdgeIt<M> edge_it(rep->order, edge_id); ~edge_it; ++edge_it)
                    {
                        const auto &pt = rep->coefs[edge_it.getFlatIndex()];
                        is_valid = _check_internal(pt);
                        if (!is_valid)
                            break;
                    }
                    add_edge = is_valid;
                }

                if (add_edge)
                    edges.push_back(rep->extractEdge(edge_id));
            }
        }

        return edges;
    }

} // namespace algoim::detail

#endif // ALGOIM_REPARAMETERIZATION_COMMON_H
