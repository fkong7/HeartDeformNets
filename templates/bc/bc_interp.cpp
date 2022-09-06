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
#include <igl/readDMAT.h>
#include <igl/readCSV.h>

#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <vector>
#include <algorithm>
#include <iostream>

#define DATA_PATH "data"
#define OUTPUT_PATH "output"
//#define DATA_PATH "/Users/fanweikong/Documents/Modeling/libigl/tutorial/data"

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
Eigen::MatrixXd W, TV;
Eigen::MatrixXi TT;
Eigen::MatrixXd source_v, W_interp, source_tv; 
Eigen::MatrixXi source_f, source_tt, source_tf;

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
        double dist_i = (centers.row(i) - pt.transpose()).squaredNorm();
        if (dist_i<dist)
        {
            index = i;
            //std::cout <<"Loop : " << index << " dist: " << dist_i << " " << dist << std::endl;
            dist = dist_i;
        }
    }
   //std::cout << "INDEX: " << index << " "  << centers.row(index) << " " << pt << " Distance: " << dist << std::endl;
   //std::cout << "Diff: " << centers.row(index) - pt << std::endl;
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
  igl::readCSV(string(argv[1]), W);
  igl::readCSV(string(argv[2]), TV);
  igl::readCSV(string(argv[3]), TT);
  cout<< "Reading volume mesh information..." << endl;
  // Interpolate a new mesh
  if (argc > 6)
  { 
      igl::readOBJ(string(argv[4]), source_v, source_f);
      cout<< "Reading interpolated mesh information..." << endl;
      igl::copyleft::tetgen::tetrahedralize(source_v, source_f, string(argv[6]), source_tv, source_tt, source_tf);
  }
  else
  {
      igl::readOBJ(string(argv[4]), source_tv, source_tt);
      cout<< "Reading interpolated mesh information..." << endl;
  }
  
  igl::AABB<Eigen::MatrixXd, 3> bbox;
  bbox.init(TV, TT);
  Eigen::VectorXi bary_index;
  igl::in_element(TV, TT, source_tv, bbox, bary_index); 
  Eigen::MatrixXd bary_coords, bary_centers, Va(bary_index.size(), 3), Vb(bary_index.size(), 3), Vc(bary_index.size(), 3), Vd(bary_index.size(), 3);
  igl::barycenter(TV, TT, bary_centers);
  cout<< "Computed bary centers..." << endl;
  int count = 0;
  for (int i=0; i<bary_index.size(); i++)
  {
      if (bary_index[i] == -1)
      {
          count++;
          bary_index[i] =  cloest_point(bary_centers, source_tv.row(i));
          //std::cout << "Outside: " << bary_centers.row(bary_index[i]) << " " << source_tv.row(i) << " dist " << (bary_centers.row(bary_index[i])-source_tv.row(i)).squaredNorm() << std::endl;
          //source_tv.row(i).array() = bary_centers.row(bary_index[i]).array();
          //source_tv.row(i).array() = TV.row(cloest_point(TV, source_tv.row(i))).array();
      }
      Va.row(i).array() = TV.row(TT.coeff(bary_index[i], 0)).array();
      Vb.row(i).array() = TV.row(TT.coeff(bary_index[i], 1)).array();
      Vc.row(i).array() = TV.row(TT.coeff(bary_index[i], 2)).array();
      Vd.row(i).array() = TV.row(TT.coeff(bary_index[i], 3)).array();
  }
  std::cout << "Number of tris outside " << count << std::endl;
  igl::barycentric_coordinates(source_tv,Va,Vb,Vc,Vd,bary_coords);
  std::cout << "barycentric_coordinates" << std::endl;
  if (argc > 6)
  {
      writeToCSVfile(string(argv[7]), source_tv);
      writeToCSVfile(string(argv[8]), source_tt);
  }
  // interpolate W based on barycentric coordinates
  igl::barycentric_interpolation(W, TT, bary_coords, bary_index, W_interp);
  
  cout<< "Interpolation finished..." << endl;
  writeToCSVfile(string(argv[5]), W_interp);
}
