#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <ransac_lib/ransac.hpp>

#include "plane_estimator.hpp"

void GenerateRandomInstance(const int num_inliers, const int num_outliers,
                            double inlier_threshold, double plane_radius,
                            const Eigen::Hyperplane<double, 3> &plane,
                            Eigen::Matrix3Xd &points) {

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
  urd angle_dist(-3.14159265358979323846, 3.14159265358979323846);

  // Generate random inlier
  Eigen::Vector3d w =
      (plane.normal().x() == 0)
          ? plane.normal().matrix().cross(Eigen::Vector3d::UnitX())
          : plane.normal().matrix().cross(Eigen::Vector3d::UnitZ());

  for (int i{0}; i < num_inliers; ++i) {
    const int index = indices[i];
    Eigen::AngleAxisd angle(angle_dist(rng), w);

    points.col(index) = angle.axis() * radius_dist(rng);

    if (plane.offset() != 0) {
      points.col(index) += plane.normal() * (plane.offset() + inlier_dist(rng));
    }
  }

  for (int i{num_inliers}; i < kNumPoints; ++i) {
    const int index = indices[i];

    points.col(index) =
        Eigen::Vector3d(radius_dist(rng), radius_dist(rng), radius_dist(rng));

    while (plane.absDistance(points.col(index)) < inlier_threshold) {
      points.col(index) =
          Eigen::Vector3d(radius_dist(rng), radius_dist(rng), radius_dist(rng));
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

  const int kNumDataPoints = 1000;
  std::vector<double> outlier_ratios = {0.1, 0.2, 0.3, 0.4,  0.5,  0.6,
                                        0.7, 0.8, 0.9, 0.95, 0.99, 0.999};
  for (const double outlier_ratio : outlier_ratios) {
    std::cout << " Inlier ratio: " << 1.0 - outlier_ratio << '\n';
    int num_outliers =
        static_cast<int>(static_cast<double>(kNumDataPoints) * outlier_ratio);
    int num_inliers = kNumDataPoints - num_outliers;

    Eigen::Matrix3Xd data;
    GenerateRandomInstance(num_inliers, num_outliers, 0.1 * 0.5, 10, plane,
                           data);

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
