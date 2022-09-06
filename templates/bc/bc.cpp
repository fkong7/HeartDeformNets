#include <igl/colon.h>
#include <igl/column_to_quats.h>
#include <igl/directed_edge_parents.h>
#include <igl/forward_kinematics.h>
#include <igl/jet.h>
#include <igl/lbs_matrix.h>
#include <igl/deform_skeleton.h>
#include <igl/normalize_row_sums.h>
#include <igl/readDMAT.h>
#include <igl/readMESH.h>
#include <igl/readTGF.h>
#include <igl/readOBJ.h>
#include <igl/AABB.h>
#include <igl/in_element.h>
#include <igl/slice.h>
#include <igl/barycentric_coordinates.h>
#include <igl/barycentric_interpolation.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/point_mesh_squared_distance.h>
#include <igl/biharmonic_coordinates.h>
#include <igl/barycenter.h>
#include <igl/decimate.h>
#include <igl/writeDMAT.h>
#include <igl/writeOBJ.h>
#include <igl/readDMAT.h>

#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <vector>
#include <algorithm>
#include <iostream>

#define DATA_PATH "data"
#define OUTPUT_PATH "output"
//#define DATA_PATH "/Users/fanweikong/Documents/Modeling/libigl/tutorial/data"
int selected = 0;
Eigen::MatrixXd V,TV,W,C,TEST_C;
Eigen::MatrixXi T,F,TT,TF,CF,TEST_CF;

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

// writing functions taking Eigen types as parameters, 
// see https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
template <typename Derived>
void writeToCSVfile(std::string name, const Eigen::MatrixBase<Derived>& matrix)
{
    std::ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
    // file.close() is not necessary, 
    // desctructur closes file, see https://en.cppreference.com/w/cpp/io/basic_ofstream
}

int cloest_point(Eigen::MatrixXd centers, Eigen::VectorXd pt)
{
    double dist = 10000;
    int index = -1;
    for (int i=0; i<centers.rows(); i++)
    {
        double dist_i = (centers.row(i) - pt).squaredNorm();
        if (dist_i<dist)
        {
            index = i;
            dist = dist_i;
        }
    }
   return index;
} 

std::string getPathName(const std::string& s) {

   char sep = '/';

#ifdef _WIN32
   sep = '\\';
#endif
   size_t i = s.rfind(sep, s.length());
   if (i != std::string::npos) {
      return(s.substr(0, i));
   }
   return("");
}


int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;
  //igl::readOBJ(getPathName(string(__FILE__))+string("/data/07_21_2021_16_27_16_template_template_full_generic_v2_manifold_simplified_tetra_001.mesh__sf.obj"), V, F);
  igl::readOBJ(string(argv[1]), V, F);
  //igl::readOBJ(DATA_PATH "/07_21_2021_16_27_16_template_template_full_generic_v2_manifold_simplified_tetra.mesh__sf.obj", V, F);
  //igl::readOBJ(DATA_PATH "/template_full_generic_v2_binary_tetra.mesh__sf.obj", V, F);
  //igl::copyleft::tetgen::tetrahedralize(V,F,"q0.5 a1e-7YV", TV,TT,TF);
  igl::copyleft::tetgen::tetrahedralize(V,F,string(argv[2]), TV,TT,TF);
  cout << "Surf verts: "<< V.rows() << " Surf faces: " << F.rows() << " Verts: " << TV.rows() << " Tets: " << TT.rows() << " Faces: " << TF.rows() << endl;
  
  igl::readOBJ(string(argv[3]),C, CF);
  Eigen::VectorXi P = Eigen::VectorXi::LinSpaced(C.rows(),0,C.rows()-1);
  // Find closest points on the tetrahedral mesh as control point handels
  Eigen::VectorXi b;
  {
    // this will create a vector from 0 to V.rows()-1 where the gap is 1
    Eigen::VectorXi J = Eigen::VectorXi::LinSpaced(TV.rows(),0,TV.rows()-1);
    Eigen::VectorXd sqrD;
    Eigen::MatrixXd _2;
    cout<<"Finding closest points..."<<endl;
    // using J which is N by 1 instead of a matrix that represents faces of N by 3
    // so that we will find the closest vertices istead of closest point on the face
    // so far the two meshes are not seperated. So what we are really doing here
    // is computing handles from low resolution and use that for the high resolution one
    igl::point_mesh_squared_distance(C,TV,J,sqrD,b,_2);
    assert(sqrD.minCoeff() < 1e-7 && "control points should exist in tetrahedral mesh.");
  }
  igl::slice(TV,b,1,C);

  // list of points --> list of singleton lists
  std::vector<std::vector<int> > S;
  // S will hav size of low.V.rows() and each list inside will have 1 element
  igl::matrix_to_list(b,S);

  cout<<"Computing weights for "<<b.size()<<
    " handles at "<<TV.rows()<<" vertices..."<<endl;
  // Technically k should equal 3 for smooth interpolation in 3d, but 2 is
  // faster and looks OK
  const int k = 3;

  igl::biharmonic_coordinates(TV,TT,S,k,W);
  writeToCSVfile(string(argv[4]), W);
  igl::writeOBJ(string(argv[5]),C, CF);
  writeToCSVfile(string(argv[6]), TV);
  writeToCSVfile(string(argv[7]), TT);
  
}
