// Examples to demonstrate Algoim's methods for reparameterizing implicitly defined domains in hyperrectangles.
// The file contains a single main() routine; compile it as you would for any .cpp file with a main() entry point.

#include "implicit_functions_lib.hpp"
#include "reparameterization_general.hpp"
#include "vtk_utils.hpp"

/**
 * @brief This test function reparameterizes the domain defined implicitly by
 * the given function @p _phi in the domain [0,1]^3.
 * A volumetric reparamterization is created for the region where @p _phi is negative.
 * The wirebasket edges of this reparameterization are also generated.
 * 
 * @tparam Phi Type of the function to reparameterize.
 * @param _phi Function to reparameterize.
 * @param _rep_order Polynomial order (degree + 1) for the reparameterization.
 * @param _fname_prefix Name prefix for generated VTK files.
 */
template<typename Phi>
void testVolume(const Phi &_phi, const int _rep_order, const std::string &_fname_prefix)
{
    const algoim::HyperRectangle<algoim::real,3> domain(0.0, 1.0);
    const auto reparam_vol = algoim::general::reparam<3, Phi>(_phi, domain, _rep_order);

    algoim::vtk::outputReparameterizationAsVTUXML(reparam_vol, _fname_prefix + "-volume");

    const auto reparam_vol_wire = algoim::general::reparamWirebasket<3, Phi>(_phi, domain, _rep_order);
    algoim::vtk::outputReparameterizationAsVTUXML(reparam_vol_wire, _fname_prefix + "-volume-wirebasket");

}

/**
 * @brief This test function reparameterizes the domain defined implicitly by
 * the given function @p _phi in the domain [0,1]^3.
 * A surface reparamterization is created for the manifold where @p _phi is zero.
 * The wirebasket edges of this reparameterization are also generated.
 * 
 * @tparam Phi Type of the function to reparameterize.
 * @param _phi Function to reparameterize.
 * @param _rep_order Polynomial order (degree + 1) for the reparameterization.
 * @param _fname_prefix Name prefix for generated VTK files.
 */
template<typename Phi>
void testSurface(const Phi &_phi, const int _rep_order, const std::string &_fname_prefix)
{
    const algoim::HyperRectangle<algoim::real,3> domain(0.0, 1.0);
    const auto reparam_srf = algoim::general::reparamLevelset<3, Phi>(_phi, domain, _rep_order);

    algoim::vtk::outputReparameterizationAsVTUXML(reparam_srf, _fname_prefix + "-surface");

    const auto reparam_srf_wire = algoim::general::reparamLevelsetWirebasket<3, Phi>(_phi, domain, _rep_order);
    algoim::vtk::outputReparameterizationAsVTUXML(reparam_srf_wire, _fname_prefix + "-surface-wirebasket");
}

/**
 * @brief This test function reparameterizes the domain defined implicitly by
 * the given function @p _phi in the domain [0,1]^3.
 * Reparameterizations are created for the traces of @p _phi in the faces the domain [0,1]^3,
 * restricted to the region where @p _phi is negative.
 * The wirebasket edges of those reparameterizations are also generated.
 * 
 * @tparam Phi Type of the function to reparameterize.
 * @param _phi Function to reparameterize.
 * @param _rep_order Polynomial order (degree + 1) for the reparameterization.
 * @param _fname_prefix Name prefix for generated VTK files.
 */
template<typename Phi>
void testFlatSurfaces(const Phi &_phi, const int _rep_order, const std::string &_fname_prefix)
{
    const algoim::HyperRectangle<algoim::real,3> domain(0.0, 1.0);
    for(int dir = 0; dir < 3; ++dir)
    {
        for(int side = 0; side < 2; ++side)
        {
            const auto reparam = algoim::general::reparamFace<3,Phi>(_phi, domain, _rep_order, dir, side);

            const std::string fname = _fname_prefix + "-face_" + std::to_string(dir) + "_" + std::to_string(side);
            algoim::vtk::outputReparameterizationAsVTUXML(reparam, fname);

            const auto reparam_wire = algoim::general::reparamFaceWirebasket<3, Phi>(_phi, domain, _rep_order, dir, side);
            algoim::vtk::outputReparameterizationAsVTUXML(reparam_wire, fname + "-wirebasket");
        }
    }
}

/**
 * @brief This test function reparameterizes the domain defined implicitly by
 * the given function @p _phi in the domain [0,1]^3.
 * Different reparameterizations are created: for the volume (the region where
 * @p _phi is negative); the levelset (the manifold where @p _phi is zero);
 * and the traces of @p _phi in the faces of the domain [0,1]^3.
 * The wirebasket edges of those reparameterizations are also generated.
 * 
 * @tparam Phi Type of the function to reparameterize.
 * @param _phi Function to reparameterize.
 * @param _rep_order Polynomial order (degree + 1) for the reparameterization.
 * @param _fname_prefix Name prefix for generated VTK files.
 */
template<typename Phi>
void test(const Phi &_phi, const int _rep_order, const std::string &_fname_prefix)
{
    testVolume(_phi, _rep_order, _fname_prefix);
    testSurface(_phi, _rep_order, _fname_prefix);
    testFlatSurfaces(_phi, _rep_order, _fname_prefix);
}

#if ALGOIM_EXAMPLES_DRIVER == 0 || ALGOIM_EXAMPLES_DRIVER == 5
int main(int argc, char* argv[])
{
    const int n_pts_dir = 3;

    {
        const algoim::Gyroid<algoim::Schoen, 3> phi;
        const std::string name = "SchoenGyroid";
        test(phi, n_pts_dir, name);
    }

    {
        const algoim::Gyroid<algoim::SchoenIWP,3> phi;
        const std::string name = "SchoenIWP";
        test(phi, n_pts_dir, name);
    }

    {
        const algoim::Gyroid<algoim::FischerKochS,3> phi;
        const std::string name = "FischerKochS";
        test(phi, n_pts_dir, name);
    }

    {
        const algoim::Gyroid<algoim::SchwarzDiamond,3> phi;
        const std::string name = "SchwarzDiamond";
        test(phi, n_pts_dir, name);
    }

    {
        const algoim::Gyroid<algoim::SchoenFRD,3> phi;
        const std::string name = "SchoenFRD";
        test(phi, n_pts_dir, name);
    }

    {
        const algoim::Gyroid<algoim::SchwarzPrimitive,3> phi;
        const std::string name = "SchwarzPrimitive";
        test(phi, n_pts_dir, name);
    }

    {
        const algoim::Sphere<3> phi(0.0, 0.9);
        const std::string name = "Sphere";
        test(phi, n_pts_dir, name);
    }

    {
        const algoim::Ellipsoid<3> phi;
        const std::string name = "Ellipsoid";
        test(phi, n_pts_dir, name);
    }

    return 0;
}
#endif
