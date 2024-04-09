#ifndef ALGOIM_EXAMPLES_VTK_UTILS_LIB_H
#define ALGOIM_EXAMPLES_VTK_UTILS_LIB_H

// Collection of tools for exporting to VTK files.

#include "uvector.hpp"
#include "real.hpp"
#include "hyperrectangle.hpp"
#include "bezier.hpp"

#include <fstream>
#include <string>

namespace algoim::vtk
{

namespace detail {

/**
 * @brief Evaluates the given function @p _phi at equidistant points.
 * 
 * @tparam Phi Type of function.
 * @tparam N Dimension of the domain.
 * @tparam R Range of the function.
 * @param _phi Function to be evaluated. The function must provided an operator() to be evaluated.
 * @param _domain Domain where the function is sampled.
 * @param _n_pts_dir Number of sampling points per direction.
 * @param _values Computed values.
 */
template<int N, int R, typename Phi>
void evalFunction(const Phi &_phi,
                  const HyperRectangle<real, N> &_domain,
                  const uvector<int, N> &_n_pts_dir,
                  std::conditional_t<R==1, std::vector<real>, std::vector<uvector<real, R>>> &_values)
{
    static_assert(N == 2 || N == 3, "Invalid dimension.");

    const auto nu = static_cast<std::size_t>(_n_pts_dir(0));
    const auto nv = static_cast<std::size_t>(_n_pts_dir(1));
    const auto nuv = nu * nv;

    const uvector<real, N> dx = _domain.extent() / (_n_pts_dir - 1);
    const auto xmin = _domain.min();

    uvector<real, N> x;

    if constexpr (N == 2)
    {
        #pragma omp parallel for private(x) collapse(2)
        for(std::size_t i = 0; i < _n_pts_dir(0); ++i)
        {
            x(0) = i * dx(0) + xmin(0);
            for(std::size_t j = 0; j < _n_pts_dir(1); ++j)
            {
                x(1) = j * dx(1) + xmin(1);
                _values[nu * j + i] = _phi(x);
            }
        }
    }
    else // if constexpr (N == 3)
    {
        const auto nw = static_cast<std::size_t>(_n_pts_dir(2));

        #pragma omp parallel for private(x) collapse(3)
        for(std::size_t i = 0; i < _n_pts_dir(0); ++i)
        {
            x(0) = i * dx(0) + xmin(0);
            for(std::size_t j = 0; j < _n_pts_dir(1); ++j)
            {
                x(1) = j * dx(1) + xmin(1);
                for(std::size_t k = 0; k < _n_pts_dir(2); ++k)
                {
                    x(2) = k * dx(2) + xmin(2);
                    _values[nuv * k + nu * j + i] = _phi(x);
                }
            }
        }
    }
}

/**
 * @brief Helper function for getting the VTK cell type for N-dimensional
 * tensor-product Bezier element of given order.
 * 
 * @tparam N Dimension of the cell.
 * @param _element Bezier element.
 * @return Cell type id.
 */
template<int N, int R>
int getCellTypeVTU(const bezier::BezierTP<N, R> &_element)
{
    if (all(_element.order == 2))
        return N == 1 ? 3 : N == 2 ? 9 : 12;
    else
        return N == 1 ? 75 : N == 2 ? 77 : 79;
}

/**
 * @brief Helper function for getting the VTK cell type for N-dimensional
 * tensor-product Lagrange element of given order.
 * 
 * @tparam N Dimension of the cell.
 * @param _element Lagrange element.
 * @return Cell type id.
 */
template<int N, int R>
int getCellTypeVTU(const lagrange::LagrangeTP<N, R> &_element)
{
    if (all(_element.order == 2))
        return N == 1 ? 3 : N == 2 ? 9 : 12;
    else
        return N == 1 ? 68 : N == 2 ? 70 : 72;
}

/**
 * @brief Helper function for creating a re-enumeration from counter-lexicographical
 * to VTK ordering for N-dimensional tensor-product cells of the given order.
 * 
 * We denote here counter-lexicographical as the ordering in which the
 * component N-1 iterates the fastest, i.e., is the inner-most, and
 * the component 0 iterates the slowest, i.e., the outer-most.
 * 
 * @tparam N Dimension of the cell.
 * @param _order Cell order along the N directions.
 * @param _mask Vector storing the generated mask.
 * @return Map from counter-lexicographical to VTK ordering. 
 */
template<int N>
void createConnectivityMaskVTU(const uvector<int, N> &_order,
                                  std::vector<std::size_t> &_mask)
{
    _mask.clear();
    _mask.reserve(prod(_order));

    if constexpr (N == 1)
    {
        // Vertices
        _mask.push_back(0);
        _mask.push_back(_order(0)-1);

        // Internal points.
        for(int u = 1; u < (_order(0)-1); ++u)
            _mask.push_back(u);
    }
    else if constexpr (N == 2)
    {
        const auto tensorToFlat = [_order](const int u, const int v)
        {
            return v + u * _order(1);
        };

        // Vertices
        _mask.push_back(tensorToFlat(0, 0));
        _mask.push_back(tensorToFlat(_order(0)-1, 0));
        _mask.push_back(tensorToFlat(_order(0)-1, _order(1)-1));
        _mask.push_back(tensorToFlat(0, _order(1)-1));

        // Edges
        for(int u = 1; u < (_order(0)-1); ++u)
            _mask.push_back(tensorToFlat(u, 0));
        for(int v = 1; v < (_order(1)-1); ++v)
            _mask.push_back(tensorToFlat(_order(0)-1, v));
        for(int u = 1; u < (_order(0)-1); ++u)
            _mask.push_back(tensorToFlat(u, _order(1)-1));
        for(int v = 1; v < (_order(1)-1); ++v)
            _mask.push_back(tensorToFlat(0, v));

        // Internal
        for(int v = 1; v < (_order(1)-1); ++v)
            for(int u = 1; u < (_order(0)-1); ++u)
                _mask.push_back(tensorToFlat(u, v));
    }
    else // if constexpr (N == 3)
    {
        const auto tensorToFlat = [_order](const int u, const int v, const int w)
        {
            return w + v * _order(2) + u * _order(1) * _order(2);
        };

        // Vertices
        _mask.push_back(tensorToFlat(0, 0, 0));
        _mask.push_back(tensorToFlat(_order(0)-1, 0, 0));
        _mask.push_back(tensorToFlat(_order(0)-1, _order(1)-1, 0));
        _mask.push_back(tensorToFlat(0, _order(1)-1, 0));
        _mask.push_back(tensorToFlat(0, 0, _order(2)-1));
        _mask.push_back(tensorToFlat(_order(0)-1, 0, _order(2)-1));
        _mask.push_back(tensorToFlat(_order(0)-1, _order(1)-1, _order(2)-1));
        _mask.push_back(tensorToFlat(0, _order(1)-1, _order(2)-1));

        // Edges
        for(int u = 1; u < (_order(0)-1); ++u)
            _mask.push_back(tensorToFlat(u, 0, 0));
        for(int v = 1; v < (_order(1)-1); ++v)
            _mask.push_back(tensorToFlat(_order(0)-1, v, 0));
        for(int u = 1; u < (_order(0)-1); ++u)
            _mask.push_back(tensorToFlat(u, _order(1)-1, 0));
        for(int v = 1; v < (_order(1)-1); ++v)
            _mask.push_back(tensorToFlat(0, v, 0));
        for(int u = 1; u < (_order(0)-1); ++u)
            _mask.push_back(tensorToFlat(u, 0, _order(2)-1));
        for(int v = 1; v < (_order(1)-1); ++v)
            _mask.push_back(tensorToFlat(_order(0)-1, v, _order(2)-1));
        for(int u = 1; u < (_order(0)-1); ++u)
            _mask.push_back(tensorToFlat(u, _order(1)-1, _order(2)-1));
        for(int v = 1; v < (_order(1)-1); ++v)
            _mask.push_back(tensorToFlat(0, v, _order(2)-1));
        for(int w = 1; w < (_order(2)-1); ++w)
            _mask.push_back(tensorToFlat(0, 0, w));
        for(int w = 1; w < (_order(2)-1); ++w)
            _mask.push_back(tensorToFlat(_order(0)-1, 0, w));
        for(int w = 1; w < (_order(2)-1); ++w)
            _mask.push_back(tensorToFlat(0, _order(1)-1, w));
        for(int w = 1; w < (_order(2)-1); ++w)
            _mask.push_back(tensorToFlat(_order(0)-1, _order(1)-1, w));

        // Faces
        for(int w = 1; w < (_order(2)-1); ++w)
            for(int v = 1; v < (_order(1)-1); ++v)
                _mask.push_back(tensorToFlat(0, v, w));
        for(int w = 1; w < (_order(2)-1); ++w)
            for(int v = 1; v < (_order(1)-1); ++v)
                _mask.push_back(tensorToFlat(_order(0)-1, v, w));
        for(int w = 1; w < (_order(2)-1); ++w)
            for(int u = 1; u < (_order(0)-1); ++u)
                _mask.push_back(tensorToFlat(u, 0, w));
        for(int w = 1; w < (_order(2)-1); ++w)
            for(int u = 1; u < (_order(0)-1); ++u)
                _mask.push_back(tensorToFlat(u, _order(1)-1, w));
        for(int v = 1; v < (_order(1)-1); ++v)
            for(int u = 1; u < (_order(0)-1); ++u)
                _mask.push_back(tensorToFlat(u, v, 0));
        for(int v = 1; v < (_order(1)-1); ++v)
            for(int u = 1; u < (_order(0)-1); ++u)
                _mask.push_back(tensorToFlat(u, v, _order(2)-1));

        // Internal
        for(int w = 1; w < (_order(2)-1); ++w)
            for(int v = 1; v < (_order(1)-1); ++v)
                for(int u = 1; u < (_order(0)-1); ++u)
                    _mask.push_back(tensorToFlat(u, v, w));
    }
}

} // detail

/**
 * @brief Writes a collection of points to a VTK XML file (.vtp).
 * 
 * @tparam N Points dimension.
 * @param _points Collection of points.
 * @param _fname File name without extension.
 */
template<int N>
void toVTPXML(const std::vector<uvector<real,N>> &_points, const std::string &_fname)
{
    std::ofstream stream(_fname + ".vtp");

    static_assert(N == 2 || N == 3, "outputQuadratureRuleAsVtpXML only supports 2D and 3D quadrature schemes");
    stream << "<?xml version=\"1.0\"?>\n";
    stream << "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    stream << "<PolyData>\n";
    stream << "<Piece NumberOfPoints=\"" << _points.size() << "\" NumberOfVerts=\"" << _points.size() << "\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n";
    stream << "<Points>\n";
    stream << "  <DataArray type=\"Float32\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">";
    for (const auto& pt : _points)
    {
        stream << pt(0) << ' ' << pt(1) << ' ';
        if (N == 3)
            stream << pt(2) << ' ';
        else
            stream << "0.0 ";
    }
    stream << "</DataArray>\n";
    stream << "</Points>\n";
    stream << "<Verts>\n";
    stream << "  <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">";
    for (std::size_t i = 0; i < _points.size(); ++i)
        stream << i << ' ';
    stream <<  "</DataArray>\n";
    stream << "  <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">";
    for (std::size_t i = 1; i <= _points.size(); ++i)
        stream << i << ' ';
    stream << "</DataArray>\n";
    stream << "</Verts>\n";
    stream << "</Piece>\n";
    stream << "</PolyData>\n";
    stream << "</VTKFile>\n";

    stream.close();
};

/**
 * @brief Writes a N-dimensional grid of values to a VTK XML file (.vtr).
 * 
 * @tparam N Number of dimensions.
 * @param _n_pts_dir Number of points per direction.
 * @param _domain N-dimensional domain of the grid of values.
 * @param _values Grid of values to be writtend stored in lexicographical order.
 * @param _fname File name without extension.
 */
template<int N = 3>
void writeStructuredRectilinearGridVTK(const uvector<int, N> _n_pts_dir, const HyperRectangle<real, N> &_domain, const std::vector<real> &_values, const std::string &_fname)
{
    if constexpr (N == 2)
    {
        const auto n_pts_dir_3D = add_component(_n_pts_dir, 2, 1);
        HyperRectangle<real, 3> domain_3D(0.0, 0.0);
        for(int dir = 0; dir < 2; ++dir)
        {
            domain_3D.min(dir) = _domain.min(dir);
            domain_3D.max(dir) = _domain.max(dir);
        }
        writeStructuredRectilinearGridVTK<3>(n_pts_dir_3D, domain_3D, _values, _fname);
    }
    else
    {

    std::ofstream f(_fname + ".vtr");

    f << "<?xml version=\"1.0\"?>\n";
    f << "<VTKFile type=\"RectilinearGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    f << "<RectilinearGrid WholeExtent=\"";
    for(int dir = 0; dir < N; ++dir)
        f << " 0 " << _n_pts_dir(dir)-1;
    f  << "\">\n";

    f << "<Piece Extent=\"";
    for(int dir = 0; dir < N; ++dir)
        f << " 0 " << _n_pts_dir(dir)-1;
    f  << "\">\n";

    f << "  <Coordinates>\n";
    for(int dir = 0; dir < N; ++dir)
    {
        const real dx = _n_pts_dir(dir) == 1 ? 0.0 : _domain.extent(dir) / (static_cast<real>(_n_pts_dir(dir)) - 1);
        const auto x0 = _domain.min(dir);

        f << "    <DataArray type=\"Float64\" format=\"ascii\">\n";
        for(int i = 0; i < _n_pts_dir(dir); ++i)
            f << " " << x0 + i * dx;
        f << "    </DataArray>\n";
    }
    f << "  </Coordinates>\n";

    f << "  <CellData>\n";
    f << "  </CellData>\n";

    f << "  <PointData>\n";
    f << "    <DataArray type=\"Float64\" Name=\"values\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    f << "      ";
    for(const auto &v : _values)
        f << v << " ";
    f << std::endl;
    f << "    </DataArray>\n";
    f << "  </PointData>\n";

    f << "</Piece>\n";
    f << "</RectilinearGrid>\n";
    f << "</VTKFile>\n";

    f.close();
    }
}

/**
 * @brief Writes a N-dimensional structured mesh to a VTK XML file (.vts).
 * 
 * @tparam N Number of dimensions.
 * @param _points Points of the mesh in lexicographical order.
 * @param _n_pts_dir Number of points per direction.
 * @param _values Values associated to the points.
 * @param _fname File name without extension.
 */
template<int N = 3>
void writeStructuredGridVTK(const std::vector<uvector<real, N>> &_points,const uvector<int, N> _n_pts_dir, const std::vector<real> &_values, const std::string &_fname)
{
    std::ofstream f(_fname + ".vts");

    f << "<?xml version=\"1.0\"?>\n";
    f << "<VTKFile type=\"StructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n";
    f << "<StructuredGrid WholeExtent=\"";
    for(int dir = 0; dir < N; ++dir)
        f << " 0 " << _n_pts_dir(dir)-1;
    f  << "\">\n";

    f << "<Piece Extent=\"";
    for(int dir = 0; dir < N; ++dir)
        f << " 0 " << _n_pts_dir(dir)-1;
    f  << "\">\n";

    f << "  <Points>\n";
    f << "    <DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"" << N << "\" format=\"ascii\">\n";

    for(const auto &pt : _points)
    {
        f << "      ";
        if constexpr (N == 3)
            f << pt(0) << ' ' << pt(1) << ' ' << pt(2) << ' ';
        else if constexpr (N == 2)
            f << pt(0) << ' ' << pt(1) << " 0.0 ";
        else
            f << pt(0) << " 0.0 0.0 ";
    }
    f << "    </DataArray>\n";
    f << "  </Points>\n";

    f << "  <CellData>\n";
    f << "  </CellData>\n";

    f << "  <PointData>\n";
    f << "    <DataArray type=\"Float64\" Name=\"values\" NumberOfComponents=\"1\" format=\"ascii\">\n";
    f << "      ";
    for(const auto &v : _values)
        f << v << " ";
    f << std::endl;
    f << "    </DataArray>\n";
    f << "  </PointData>\n";

    f << "</Piece>\n";
    f << "</StructuredGrid>\n";
    f << "</VTKFile>\n";

    f.close();
}

/**
 * @brief Writes a collection of tensor-product elements to a VTK XML file (.vtu).
 * 
 * @tparam E Element type (either BezierTP or LagrangeTP).
 * @param _elements Elements to be written.
 * @param _fname File name without extension.
 */
template<typename E>
void outputReparameterizationAsVTUXML(const std::vector<std::shared_ptr<E>> &_elements, const std::string &_fname)
{
    static const int N = E::N;
    static const int R = E::R;

    static_assert(1 <= R && R <= 3 && 1 <= N && N <= 3,
                  "outputReparameterizationAsVTUXML only supports 1D, 2D, and 3D cells.");

    const auto nCells = _elements.size();

    std::size_t nPts = 0;
    for(const auto &elem : _elements)
    {
        assert(elem != nullptr);
        nPts += elem->coefs.size();
    }

    std::ofstream stream(_fname + ".vtu");

    stream << "<?xml version=\"1.0\"?>\n";
    stream << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    stream << "<UnstructuredGrid>\n";
    stream << "<Piece NumberOfPoints=\"" << nPts << "\" NumberOfCells=\"" << nCells << "\">\n";
    stream << "<Points>\n";
    stream << "  <DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">";

    for(const auto &elem : _elements)
    {
        for(const auto &pt : elem->coefs)
        {
            if constexpr (R == 3)
              stream << pt(0) << ' ' << pt(1) << ' ' << pt(2) << ' ';
            else if constexpr (R == 2)
              stream << pt(0) << ' ' << pt(1) << " 0.0 ";
            else
              stream << pt(0) << " 0.0 0.0 ";
        }
    }
    stream << "</DataArray>\n";
    stream << "</Points>\n";

    stream << "<Cells>\n";
    stream << "  <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">";
    std::size_t offset = 0;
    std::vector<std::size_t> offsets;
    offsets.reserve(_elements.size());

    std::vector<std::size_t> mask;
    uvector<int, N> order(-1);

    for(const auto &elem : _elements)
    {
        if (any(order != elem->order))
        {
            detail::createConnectivityMaskVTU(elem->order, mask);
            order = elem->order;
        }

        for(const auto c : mask)
          stream << offset + c << ' ';
        offset += elem->coefs.size();
        offsets.push_back(offset);
    }
    stream <<  "</DataArray>\n";

    stream << "  <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">";
    for(const auto offset : offsets)
        stream << offset << ' ';
    stream << "</DataArray>\n";

    stream << "  <DataArray type=\"Int32\" Name=\"types\" format=\"ascii\">";
    for(const auto &elem : _elements)
    {
        stream << detail::getCellTypeVTU(*elem) << ' ';
    }
    stream << "</DataArray>\n";

    stream << "  <DataArray type=\"Int32\" Name=\"HIGHERORDERDEGREES\" NumberOfComponents=\"3\" format=\"ascii\">";
    for(const auto &elem : _elements)
    {
        const auto &order = elem->order;
        for(int dir = 0; dir < N; ++dir)
            stream << order(dir) << ' ';
        for(int dir = N; dir < 3; ++dir)
            stream << 1 << ' ';
    }
    stream << "</DataArray>\n";

    stream << "</Cells>\n";

    stream << "</Piece>\n";
    stream << "</UnstructuredGrid>\n";
    stream << "</VTKFile>\n";

    stream.close();
};


/**
 * @brief Writes a VTK .vtmb file including a collection of other VTK files.
 * 
 * The files to be included must be named as <_fname_prefix>_X.<_file_ext>
 * where X is the index (cardinal) of the file.
 * 
 * The generated file is named <_fname_prefix>.vtmb
 * 
 * @param _fname_prefix Prefix of the files to include.
 * @param _file_ext Extension of the files to include.
 * @param _folder Folder where file is placed.
 * @param _i0 Index of the first file to include.
 * @param _i1 Index of the last file to include.
 */
inline void createVTMBFile(const std::string &_fname_prefix, const std::string &_file_ext, const std::string &_folder, const int _i0, const int _i1)
{
    std::ofstream f(_folder + "/" + _fname_prefix + ".vtmb");

    f << "<?xml version=\"1.0\"?>\n";
    f << "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    f << "  <vtkMultiBlockDataSet>\n";
    for(int i = _i0; i <= _i1; ++i)
    {
        f << "    <DataSet index=\"" << i << "\" name=\"" << _fname_prefix << "_" << i;
        f <<  "\" file=\"" << _fname_prefix << "_" << i << "." << _file_ext << "\"/>\n";
    }
    f << "  </vtkMultiBlockDataSet>\n";
    f << "</VTKFile>\n";

    f.close();
}

/**
 * @brief Samples a function and writes the value to a VTK file (.vtr).
 * 
 * @tparam Phi Type of the function to evaluate.
 * @tparam N Parametric domain of the function.
 * @param _phi Function to evaluate.
 * @param _domain Domain where the function is evaluated.
 * @param _n_vtk_pts_dir Number of evaluation points per direction in @p _domain.
 * @param _fname_prefix Prefix of the file name to be generated.
 */
template<typename Phi, int N>
void writeFunctionToVTK(const Phi &_phi,
                     const HyperRectangle<real, N> &_domain,
                     const uvector<int, N> &_n_vtk_pts_dir,
                     const std::string &_fname_prefix)
{
    std::size_t n_pts = 1;
    for(int dir = 0; dir < N; ++dir)
        n_pts *= _n_vtk_pts_dir(dir);

    std::vector<real> values(n_pts);
    detail::evalFunction<N,1>(_phi, _domain, _n_vtk_pts_dir, values);
    writeStructuredRectilinearGridVTK<N>(_n_vtk_pts_dir, _domain, values, _fname_prefix);
}

} // namespace algoim::vtk


#endif // ALGOIM_EXAMPLES_VTK_UTILS_LIB_H