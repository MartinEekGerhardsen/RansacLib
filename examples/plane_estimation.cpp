#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <memory> 
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <ransac_lib/ransac.hpp>

#include "plane_estimator.hpp"

void GenerateRandomInstance(const int num_inliers, const int num_outliers,
                            double inlier_threshold, double plane_radius,
                            const Eigen::Hyperplane<double, 3> &plane,
                            Eigen::Matrix3Xd &points, 
                            Eigen::ArrayXi& inliers, 
                            Eigen::ArrayXi& outliers) {
  inliers.resize(num_inliers); 
  outliers.resize(num_outliers); 

  const int kNumPoints = num_inliers + num_outliers;
  points.resize(3, kNumPoints);

  std::vector<int> indices(kNumPoints);
  std::iota(indices.begin(), indices.end(), 0);

  std::random_device device;
  std::mt19937 rng(device());

  std::shuffle(indices.begin(), indices.end(), rng);

  using urd = std::uniform_real_distribution<double>;

  urd inlier_dist(-inlier_threshold, inlier_threshold);
  urd radius_dist(inlier_threshold, plane_radius + inlier_threshold);
  urd outlier_radius_dist(-plane_radius - inlier_threshold, plane_radius + inlier_threshold); 
  urd angle_dist(-3.14159265358979323846, 3.14159265358979323846);
 
  // Planar point closest to origin
  Eigen::Vector3d centre = plane.normal() * plane.offset() / plane.normal().squaredNorm(); 

  // Create spherical representation of normal 
  double radius_orthogonal = plane.normal().norm(); 
  double phi_orthogonal = std::atan2(plane.normal().y(), plane.normal().x()); 
  double theta_orthogonal = std::fmod(
    std::atan(
      (plane.normal().template topRows<2>().norm())
      / (plane.normal().z())
    ) + (3.14159265358979323846 / 2), 
    3.14159265358979323846
  );

  Eigen::Vector3d orthogonal; 
  orthogonal 
    <<  radius_orthogonal * std::cos(phi_orthogonal) * std::sin(theta_orthogonal), 
        radius_orthogonal * std::sin(phi_orthogonal) * std::sin(theta_orthogonal), 
        radius_orthogonal * std::cos(theta_orthogonal); 

  for (int i{0}; i < num_inliers; ++i) {
    const int index = indices[i];
    inliers(i) = index; 

    double radius = radius_dist(rng); 
    double angle = angle_dist(rng); 

    points.col(index) = centre + radius * (
      Eigen::AngleAxisd(angle, plane.normal()) * orthogonal
    ); 
  }

  for (int i{num_inliers}; i < kNumPoints; ++i) {
    const int index = indices[i];
    outliers(i - num_inliers) = index; 

    points.col(index) = centre + Eigen::Vector3d::NullaryExpr(
      [&]() {return outlier_radius_dist(rng); }
    );

    while (plane.absDistance(points.col(index)) < inlier_threshold) {
      points.col(index) = centre + Eigen::Vector3d::NullaryExpr(
        [&]() {return outlier_radius_dist(rng); }
      );
    }
  }
}

int main(int argc, char **argv) {
  ransac_lib::LORansacOptions options;
  options.min_num_iterations_ = 100;
  options.max_num_iterations_ = 100000;
  options.squared_inlier_threshold_ = 0.1 * 0.1;

  std::random_device device;
  options.random_seed_ = device();

  Eigen::Hyperplane<double, 3> plane(Eigen::Vector3d(0.5, 0, 0.5),
                                     Eigen::Vector3d(10, 12, -1));
  plane.normalize(); 

  const int kNumDataPoints = 1000;
  std::vector<double> outlier_ratios = {0.1, 0.2, 0.3, 0.4,  0.5,  0.6,
                                        0.7, 0.8, 0.9, 0.95, 0.99, 0.999};
  for (const double outlier_ratio : outlier_ratios) {
    std::cout << " Inlier ratio: " << 1.0 - outlier_ratio << '\n';
    int num_outliers =
        static_cast<int>(static_cast<double>(kNumDataPoints) * outlier_ratio);
    int num_inliers = kNumDataPoints - num_outliers;

    Eigen::Matrix3Xd data;
    Eigen::ArrayXi inliers, outliers; 
    GenerateRandomInstance(num_inliers, num_outliers, 0.1 * 0.5, 10, plane,
                           data, inliers, outliers);

    // {
    //   Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n"); 
      
    //   std::stringstream ss_data, ss_inliers, ss_outliers; 
      
    //   std::string path = "/home/martin/dev/RansacLib/examples/data/"; 
    //   ss_data << path << "data_" << outlier_ratio << ".csv"; 
    //   ss_inliers << path << "inliers_" << outlier_ratio << ".csv"; 
    //   ss_outliers << path << "outliers_" << outlier_ratio << ".csv"; 
      
    //   std::ofstream data_file; 
    //   data_file.open(ss_data.str().c_str()); 
    //   std::ofstream inliers_file; 
    //   inliers_file.open(ss_inliers.str().c_str()); 
    //   std::ofstream outliers_file; 
    //   outliers_file.open(ss_outliers.str().c_str()); 

    //   data_file << data.format(CSVFormat); 
    //   inliers_file << inliers.format(CSVFormat); 
    //   outliers_file << outliers.format(CSVFormat); 
    // }

    ransac_lib::PlaneEstimator solver(data);
    ransac_lib::LocallyOptimizedMSAC<Eigen::Hyperplane<double, 3>,
                                     std::vector<Eigen::Hyperplane<double, 3>>,
                                     ransac_lib::PlaneEstimator>
        lomsac;

    ransac_lib::RansacStatistics ransac_stats;
    std::cout << "   ... running LOMSAC\n";

    Eigen::Hyperplane<double, 3> best_model;
    int num_ransac_inliers =
        lomsac.EstimateModel(options, solver, &best_model, &ransac_stats);

    std::cout << "    ... LOMSAC found " << num_ransac_inliers << " inliers in "
              << ransac_stats.num_iterations
              << " iterations with an inlier ratio of "
              << ransac_stats.inlier_ratio << '\n';
  }

  return 0;
}
