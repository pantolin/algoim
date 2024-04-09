// Examples to demonstrate Algoim's methods for reparameterizing implicitly defined domains in hyperrectangles through Beziers.
// The file contains a single main() routine; compile it as you would for any .cpp file with a main() entry point.

#include "real.hpp"
#include "bezier.hpp"
#include "hyperrectangle.hpp"
#include "reparameterization_bezier.hpp"
#include "vtk_utils.hpp"
#include "implicit_functions_lib.hpp"

#include <memory>


// 2D function corresponding to Figure 3, row 3, left column,
// https://doi.org/10.1016/j.jcp.2021.110720
static algoim::real func_0(const algoim::uvector<algoim::real,2>& _xx)
{
    const algoim::real x = _xx(0)*2 - 1;
    const algoim::real y = _xx(1)*2 - 1;
    return -0.06225100787918392 + 0.1586472897571363*y + 0.5487135634635731*y*y + 
        x*(0.3478849533965025 - 0.3321074999999999*y - 0.5595163485848738*y*y) + 
        x*x*(0.7031095851739786 + 0.29459557349175747*y + 0.030425624999999998*y*y);
}


// 3D function corresponding to Figure 3, row 3, right column,
// https://doi.org/10.1016/j.jcp.2021.110720
static algoim::real func_1(const algoim::uvector<algoim::real,3>& _xx)
{
    const algoim::real x = _xx(0)*2 - 1;
    const algoim::real y = _xx(1)*2 - 1;
    const algoim::real z = _xx(2)*2 - 1;
    return -0.3003521613375472 - 0.22416584292513722*z + 0.07904600284034838*z*z +
        y*(-0.022501556528537706 - 0.16299445153615613*z - 0.10968042065096766*z*z) + 
        y*y*(0.09321375574517882 - 0.07409794846221623*z + 0.09940785133211516*z*z) + 
        x*(0.094131400740032 - 0.11906280402685224*z - 0.010060302873268541*z*z + 
        y*y*(0.01448948481714108 - 0.0262370580373332*z - 0.08632912757566019*z*z) + 
        y*(0.08171132326327647 - 0.09286444275596013*z - 0.07651000354823911*z*z)) + 
        x*x*(-0.0914370528387867 + 0.09778971384044874*z - 0.1086777644685091*z*z + 
        y*y*(-0.04283439400630859 + 0.0750156999192893*z + 0.051754527934553866*z*z) + 
        y*(-0.052642188754328405 - 0.03538476045586772*z + 0.11117016852276898*z*z));
}

// First function of 2D implicitly-defined domain involving the intersection of two polynomials.
// This example corresponds to the top-left example of Figure 15, https://doi.org/10.1016/j.jcp.2021.110720
static algoim::real func_2_0(const algoim::uvector<algoim::real,2>& _xx)
{
    const algoim::real x = _xx(0)*2 - 1;
    const algoim::real y = _xx(1)*2 - 1;
    return 0.014836540349115947 + 0.7022484024095262*y + 0.09974561176434385*y*y +
        x*(0.6863910464417281 + 0.03805619999999999*y - 0.09440658332756446*y*y) + 
        x*x*(0.19266932968830816 - 0.2325190091204104*y + 0.2957473125000001*y*y);
}

// Second function of 2D implicitly-defined domain involving the intersection of two polynomials.
// This example corresponds to the top-left example of Figure 15, https://doi.org/10.1016/j.jcp.2021.110720
static algoim::real func_2_1(const algoim::uvector<algoim::real,2>& _xx)
{
    const algoim::real x = _xx(0)*2 - 1;
    const algoim::real y = _xx(1)*2 - 1;
    return -0.18792528379702625 + 0.6713882473904913*y + 0.3778666084723582*y*y +
        x*x*(-0.14480813208127946 + 0.0897755603159206*y - 0.141199875*y*y) + 
        x*(-0.6169311810674598 - 0.19449299999999994*y - 0.005459163675646665*y*y);
}

/**
 * @brief This test function reparameterizes the domain defined implicitly by
 * the given function @p _phi. The domain is defined as the region where
 * @p _phi is negative.
 * The function is first approximated as a polynomial.
 * 
 * Different reparameterizations are created: for the volume (the region where
 * @p _phi is negative); the levelset (the manifold where @p _phi is zero);
 * and the wirebasket edges of those reparameterizations.
 * 
 * @tparam N Parametric domain.
 * @param _phi Function to reparameterize.
 * @param _xmin Minimum boundary of reparameterization domain.
 * @param _xmax Maximum boundary of reparameterization domain.
 * @param _pol_order Polynomial order (degree + 1) for the approximation.
 * @param _rep_order Polynomial order (degree + 1) for the reparameterization.
 * @param _fname_prefix Name prefix for generated VTK files.
 */
template<int N>
void test(const std::function<algoim::real(const algoim::uvector<algoim::real,N>&)> &_phi, 
          const algoim::real _xmin,
          const algoim::real _xmax,
          const int _pol_order,
          const int _rep_order,
          const std::string &_fname_prefix)
{
    // Construct bzr by mapping [0,1] onto bounding box [xmin,xmax]
    const auto bzr = std::make_shared<algoim::bezier::BezierTP<N,1>>(_pol_order);
    algoim::xarray<algoim::real,N> alpha(bzr->coefs.data(), _pol_order);
    algoim::bernstein::bernsteinInterpolate<N>(
        [&](const algoim::uvector<algoim::real,N>& _x)
        {
            return _phi(_xmin + _x * (_xmax - _xmin));
        }, alpha);

    const auto reparams = algoim::bezier::reparam<N, false>(bzr->getXarray(), _rep_order);
    algoim::vtk::outputReparameterizationAsVTUXML(reparams, _fname_prefix + "_vol");

    const auto reparams_wires = algoim::bezier::reparamWirebasket<N, false>(bzr->getXarray(), _rep_order);
    algoim::vtk::outputReparameterizationAsVTUXML(reparams_wires, _fname_prefix + "_vol_wirebasket");

    const auto reparams_srf = algoim::bezier::reparam<N, true>(bzr->getXarray(), _rep_order);
    algoim::vtk::outputReparameterizationAsVTUXML(reparams_srf, _fname_prefix + "_srf");

    if constexpr (N == 3)
    {
        const auto reparams_srf_wires = algoim::bezier::reparamWirebasket<N, true>(bzr->getXarray(), _rep_order);
        algoim::vtk::outputReparameterizationAsVTUXML(reparams_srf_wires, _fname_prefix + "_srf_wirebasket");
    }
}

/**
 * @brief This test function reparameterizes the domain defined implicitly by
 * the given functions @p _phi0 and  @p _phi1. The domain is defined as the region where
 * both @p _phi0 and @p _phi1 are negative.
 * The functions are first approximated as a polynomials.
 * 
 * Different reparameterizations are created: for the volume (the region where
 * @p _phi0 and @ _phi1 are negative at the same time);
 * the levelset (the manifold where either @p _phi0 or @p _phi1 are zero);
 * and the wirebasket edges of those reparameterizations.
 * 
 * @tparam N Parametric domain.
 * @param _phi0 First function to reparameterize.
 * @param _phi1 Second function to reparameterize.
 * @param _xmin Minimum boundary of reparameterization domain.
 * @param _xmax Maximum boundary of reparameterization domain.
 * @param _pol_order Polynomial order (degree + 1) for the approximation.
 * @param _rep_order Polynomial order (degree + 1) for the reparameterization.
 * @param _fname_prefix Name prefix for generated VTK file.
 */
template<int N>
void test(const std::function<algoim::real(const algoim::uvector<algoim::real,N>&)> &_phi0, 
          const std::function<algoim::real(const algoim::uvector<algoim::real,N>&)> &_phi1, 
          const algoim::real _xmin,
          const algoim::real _xmax,
          const int _pol_order,
          const int _rep_order,
          const std::string &_fname_prefix)
{
    // Construct bzr0 and bzr1 by mapping [0,1] onto bounding box [xmin,xmax]
    const auto bzr0 = std::make_shared<algoim::bezier::BezierTP<N,1>>(_pol_order);
    algoim::xarray<algoim::real,N> alpha0(bzr0->coefs.data(), _pol_order);
    algoim::bernstein::bernsteinInterpolate<N>(
        [&](const algoim::uvector<algoim::real,N>& _x)
        {
            return _phi0(_xmin + _x * (_xmax - _xmin));
        }, alpha0);

    const auto bzr1 = std::make_shared<algoim::bezier::BezierTP<N,1>>(_pol_order);
    algoim::xarray<algoim::real,N> alpha1(bzr1->coefs.data(), _pol_order);
    algoim::bernstein::bernsteinInterpolate<N>(
        [&](const algoim::uvector<algoim::real,N>& _x)
        {
            return _phi1(_xmin + _x * (_xmax - _xmin));
        }, alpha1);

    const auto reparams = algoim::bezier::reparam<N, false>(bzr0->getXarray(), bzr1->getXarray(), _rep_order);
    algoim::vtk::outputReparameterizationAsVTUXML(reparams, _fname_prefix + "_vol");

    const auto reparams_wires = algoim::bezier::reparamWirebasket<N, false>(bzr0->getXarray(), bzr1->getXarray(), _rep_order);
    algoim::vtk::outputReparameterizationAsVTUXML(reparams_wires, _fname_prefix + "_vol_wirebasket");

    const auto reparams_srf = algoim::bezier::reparam<N, true>(bzr0->getXarray(), bzr1->getXarray(), _rep_order);
    algoim::vtk::outputReparameterizationAsVTUXML(reparams_srf, _fname_prefix + "_srf");

    if constexpr (N == 3)
    {
        const auto reparams_srf_wires = algoim::bezier::reparamWirebasket<N, true>(bzr0->getXarray(), bzr1->getXarray(), _rep_order);
        algoim::vtk::outputReparameterizationAsVTUXML(reparams_srf, _fname_prefix + "_srf_wirebasket");
    }
}


#if ALGOIM_EXAMPLES_DRIVER == 0 || ALGOIM_EXAMPLES_DRIVER == 6
int main(int argc, char* argv[])
{
    const int pol_order{3};
    const int rep_order{7};

    const algoim::Ellipsoid<2> phi_0;
    test<2>(phi_0, -1.1, 1.1, pol_order, rep_order, "test_0");

    const algoim::Ellipsoid<3> phi_1;
    test<3>(phi_1, -1.1, 1.1, pol_order, rep_order, "test_1");

    const auto phi_2 = func_0;
    test<2>(phi_2, 0.0, 1.0, pol_order, rep_order, "test_2");

    const auto phi_3 = func_1;
    test<3>(phi_3, 0.0, 1.0, pol_order, rep_order, "test_3");

    const auto phi_4_0 = func_2_0;
    const auto phi_4_1 = func_2_1;
    test<2>(phi_4_0, phi_4_1, 0.0, 1.0, pol_order, rep_order, "test_4");

    return 0;
}
#endif
