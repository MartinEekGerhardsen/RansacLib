#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include <ransac_lib/ransac.hpp>

#include <matplot/backend/opengl.h>
#include <matplot/matplot.h>

#include "plane_estimator.hpp"

template <class Scalar> class uniform_root_real_distribution {
public:
  uniform_root_real_distribution(Scalar a, Scalar b) : a_{a}, b_{b} {}

  template <class Generator> Scalar operator()(Generator &g) {
    std::uniform_real_distribution<Scalar> uniform(0.0, 1.0);

    return std::sqrt(uniform(g)) * (b_ - a_) + a_;
  }

private:
  Scalar a_, b_;
};

template <class Scalar, int Dim>
Scalar solve_axis(const Eigen::Hyperplane<Scalar, Dim> &plane,
                  const std::vector<Scalar> &data, int axis,
                  const std::vector<int> &other_axes) {

  double result{0.0};

  for (int other : other_axes)
    result -= plane.normal()(other) * data[other];

  result -= plane.offset();
  result /= plane.normal()(axis);

  return result;
}

template <class DerivedCartesian, class DerivedSpherical>
void cartesian2spherical(const Eigen::MatrixBase<DerivedCartesian> &cartesian,
                         Eigen::MatrixBase<DerivedSpherical> &spherical) {

  spherical.resize(3, cartesian.cols());

  spherical.row(0) = cartesian.colwise().norm();
  spherical.row(1).array() =
      (cartesian.row(2).array() / spherical.row(0).array()).acos();
  spherical.row(2) = Eigen::VectorXd::NullaryExpr(
      cartesian.cols(), [&cartesian](const Eigen::Index i) {
        return std::atan2(cartesian(1, i), cartesian(0, i));
      });
}

template <class DerivedSpherical, class DerivedCartesian>
void spherical2cartesian(const Eigen::MatrixBase<DerivedSpherical> &spherical,
                         Eigen::MatrixBase<DerivedCartesian> &cartesian) {
  cartesian.resize(3, spherical.cols());

  cartesian.row(0).array() = spherical.row(0).array() *
                             spherical.row(1).array().sin() *
                             spherical.row(2).array().cos();
  cartesian.row(1).array() = spherical.row(0).array() *
                             spherical.row(1).array().sin() *
                             spherical.row(2).array().sin();
  cartesian.row(2).array() =
      spherical.row(0).array() * spherical.row(1).array().cos();
}

template <class Scalar, int Dim>
void closestToPlane(const Eigen::Hyperplane<Scalar, Dim> &plane,
                    Eigen::Matrix<Scalar, Dim, 1> &point) {
  point = -plane.normal() * plane.offset() / plane.normal().squaredNorm();
}

template <class Scalar, int Dim>
void closestToPlane(const Eigen::Hyperplane<Scalar, Dim> &plane,
                    const Eigen::Matrix<Scalar, Dim, 1> &origin,
                    Eigen::Matrix<Scalar, Dim, 1> &point) {
  const Eigen::Matrix<Scalar, Dim, 1> shifted_normal = plane.normal() - origin;
  const Scalar shifted_offset = plane.offset() - shifted_normal.dot(origin);

  Eigen::Hyperplane<Scalar, Dim> shifted(shifted_normal, shifted_offset);

  closestToPlane(shifted, point);
}

struct plane_handle {
  Eigen::Hyperplane<double, 3> plane_;

  matplot::line_handle normal_;
  matplot::surface_handle surface_;
};

plane_handle plotPlane(matplot::axes_handle &ax,
                       const Eigen::Hyperplane<double, 3> &plane) {
  plane_handle handle;
  handle.plane_ = plane;

  // plot normal vector
  Eigen::Vector3d centre;
  closestToPlane(handle.plane_, centre);

  std::vector<double> x{centre.x(), centre.x() + plane.normal().x()};
  std::vector<double> y{centre.y(), centre.y() + plane.normal().y()};
  std::vector<double> z{centre.z(), centre.z() + plane.normal().z()};

  handle.normal_ = ax->plot3(x, y, z);

  // plot surface
  Eigen::Index maxAxis;
  handle.plane_.normal().array().abs().maxCoeff(&maxAxis);

  std::vector<int> other_axes(3 - 1, -1);
  auto inc = [](int &n, int count) {
    n += count - 1;
    return n++;
  };
  std::generate(other_axes.begin(), other_axes.end(), [&, n = 0]() mutable {
    return (n != maxAxis) ? n++ : inc(n, 2);
  });

  std::vector<double> data(3, 0.0);

  std::vector<double> other_base_0 = matplot::linspace(
      centre(other_axes[0]) - 0.5, centre(other_axes[0]) + 0.5, 10);
  std::vector<double> other_base_1 = matplot::linspace(
      centre(other_axes[1]) - 0.5, centre(other_axes[1]) + 0.5, 10);

  auto [other_0, other_1] = matplot::meshgrid(other_base_0, other_base_1);
  auto primary =
      matplot::transform(other_0, other_1, [&](double o_0, double o_1) {
        data[other_axes[0]] = o_0;
        data[other_axes[1]] = o_1;
        return solve_axis(handle.plane_, data, maxAxis, other_axes);
      });

  std::map<int, matplot::vector_2d> joint{
      {maxAxis, primary}, {other_axes[0], other_0}, {other_axes[1], other_1}};

  handle.surface_ = matplot::surf(joint[0], joint[1], joint[2]);

  return handle;
}

void GenerateRandomInstance(const int num_inliers, const int num_outliers,
                            double inlier_threshold, double plane_radius,
                            const Eigen::Hyperplane<double, 3> &plane,
                            Eigen::Matrix3Xd &points, Eigen::ArrayXi &inliers,
                            Eigen::ArrayXi &outliers) {
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
  using urrd = uniform_root_real_distribution<double>;

  urd inlier_dist(-inlier_threshold, inlier_threshold);
  urrd radius_dist(inlier_threshold, plane_radius + inlier_threshold);
  urd outlier_radius_dist(-plane_radius - inlier_threshold,
                          plane_radius + inlier_threshold);
  urd angle_dist(-3.14159265358979323846, 3.14159265358979323846);

  // Planar point closest to origin
  Eigen::Vector3d centre;
  closestToPlane(plane, centre);

  // Create spherical representation of normal
  Eigen::Vector3d spherical_orthogonal;
  cartesian2spherical(plane.normal(), spherical_orthogonal);

  spherical_orthogonal(1) =
      std::fmod(spherical_orthogonal(1) + 3.14159265358979323846 / 2.0,
                3.14159265358979323846);

  Eigen::Vector3d orthogonal;
  spherical2cartesian(spherical_orthogonal, orthogonal);

  for (int i{0}; i < num_inliers; ++i) {
    const int index = indices[i];
    inliers(i) = index;

    double radius = radius_dist(rng);
    double angle = angle_dist(rng);

    points.col(index) =
        centre +
        radius * (Eigen::AngleAxisd(angle, plane.normal()) * orthogonal);
  }

  auto random_outlier_radius = [&]() { return outlier_radius_dist(rng); };
  for (int i{num_inliers}; i < kNumPoints; ++i) {
    const int index = indices[i];
    outliers(i - num_inliers) = index;

    points.col(index) =
        centre + Eigen::Vector3d::NullaryExpr(random_outlier_radius);

    while (plane.absDistance(points.col(index)) < inlier_threshold) {
      points.col(index) =
          centre + Eigen::Vector3d::NullaryExpr(random_outlier_radius);
    }
  }
}

int main(int argc, char **argv) {
  ransac_lib::LORansacOptions options;
  options.min_num_iterations_ = 100;
  options.max_num_iterations_ = 100'000;
  options.squared_inlier_threshold_ = 0.1 * 0.1;

  std::random_device device;
  options.random_seed_ = device();

  // Eigen::Hyperplane<double, 3> plane(Eigen::Vector3d(1, 1, 1), 0);
  Eigen::Hyperplane<double, 3> plane(Eigen::Vector3d(1, 1, 1),
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
    //   Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols,
    //   ", ", "\n");

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
    best_model.normalize();

    std::cout << "    ... LOMSAC found " << num_ransac_inliers << " inliers in "
              << ransac_stats.num_iterations
              << " iterations with an inlier ratio of "
              << ransac_stats.inlier_ratio << '\n'
              << "    ... and a score of " << ransac_stats.best_model_score
              << '\n';

    std::cout << "Original plane normal: " << plane.normal().transpose()
              << "\nOriginal plane offset: " << plane.offset() << '\n';
    std::cout << "RANSAC   plane normal: " << best_model.normal().transpose()
              << "\nRANSAC   plane offset: " << plane.offset() << '\n';

    double gt_inliers_distance{0}, gt_ransac_distance{0};
    double ransac_inliers_distance{0}, ransac_ransac_distance{0};
    for (int i{0}; i < inliers.size(); ++i) {
      gt_inliers_distance += plane.absDistance(data.col(inliers(i)));
      ransac_inliers_distance += best_model.absDistance(data.col(inliers(i)));
    }

    for (int i{0}; i < ransac_stats.inlier_indices.size(); ++i) {
      gt_ransac_distance +=
          plane.absDistance(data.col(ransac_stats.inlier_indices[i]));
      ransac_ransac_distance +=
          best_model.absDistance(data.col(ransac_stats.inlier_indices[i]));
    }

    std::cout << "\nUsing GT inliers: \nGT normal: " << gt_inliers_distance
              << "\nRANSAC normal: " << ransac_inliers_distance << '\n';

    std::cout << "Using RANSAC inliers: \nGT normal: " << gt_ransac_distance
              << "\nRANSAC normal: " << ransac_ransac_distance << "\n\n";

    auto fig1 = matplot::figure(false);
    auto ax1 = fig1->add_axes();
    // auto fig = matplot::figure(false);
    // auto ax1 = fig->add_subplot(1, 2, 0);

    ax1->clear();
    ax1->hold(true);
    plotPlane(ax1, plane);

    // std::vector<double> x(data.cols()), y(data.cols()), z(data.cols());
    // Eigen::Map<Eigen::VectorXd>(x.data(), data.cols()) = data.row(0);
    // Eigen::Map<Eigen::VectorXd>(y.data(), data.cols()) = data.row(1);
    // Eigen::Map<Eigen::VectorXd>(z.data(), data.cols()) = data.row(2);
    // ax1->scatter3(x, y, z);

    Eigen::Index n_in{inliers.size()}, n_out{data.cols() - inliers.size()};
    std::vector<double> x_gt_inliers(n_in), y_gt_inliers(n_in),
        z_gt_inliers(n_in);
    std::vector<double> x_gt_outliers(n_out), y_gt_outliers(n_out),
        z_gt_outliers(n_out);

    for (int i{0}; i < inliers.size(); ++i) {
      x_gt_inliers[i] = data(0, inliers(i));
      y_gt_inliers[i] = data(1, inliers(i));
      z_gt_inliers[i] = data(2, inliers(i));
    }

    for (int i{0}; i < outliers.size(); ++i) {
      x_gt_outliers[i] = data(0, outliers(i));
      y_gt_outliers[i] = data(1, outliers(i));
      z_gt_outliers[i] = data(2, outliers(i));
    }

    ax1->scatter3(x_gt_inliers, y_gt_inliers, z_gt_inliers);
    ax1->scatter3(x_gt_outliers, y_gt_outliers, z_gt_outliers);

    ax1->hold(false);
    ax1->axis(matplot::equal);

    auto fig2 = matplot::figure(false);
    auto ax2 = fig2->add_axes();
    // auto ax2 = fig->add_subplot(1, 2, 1);

    ax2->hold(true);

    plotPlane(ax2, best_model);
    std::vector<double> x_ransac_inliers, y_ransac_inliers, z_ransac_inliers;
    x_ransac_inliers.reserve(ransac_stats.inlier_indices.size());
    y_ransac_inliers.reserve(ransac_stats.inlier_indices.size());
    z_ransac_inliers.reserve(ransac_stats.inlier_indices.size());
    std::vector<double> x_ransac_outliers, y_ransac_outliers, z_ransac_outliers;
    x_ransac_outliers.reserve(data.cols() - ransac_stats.inlier_indices.size());
    y_ransac_outliers.reserve(data.cols() - ransac_stats.inlier_indices.size());
    z_ransac_outliers.reserve(data.cols() - ransac_stats.inlier_indices.size());
    // ax2->scatter3(x, y, z);

    for (int i{0}, j{0}; i < data.cols(); ++i) {
      if (i == ransac_stats.inlier_indices[j]) {
        x_ransac_inliers.push_back(data(0, i));
        y_ransac_inliers.push_back(data(1, i));
        z_ransac_inliers.push_back(data(2, i));
        ++j;
      } else {
        x_ransac_outliers.push_back(data(0, i));
        y_ransac_outliers.push_back(data(1, i));
        z_ransac_outliers.push_back(data(2, i));
      }
    }

    ax2->scatter3(x_ransac_inliers, y_ransac_inliers, z_ransac_inliers);
    ax2->scatter3(x_ransac_outliers, y_ransac_outliers, z_ransac_outliers);

    ax2->hold(false);
    ax2->axis(matplot::equal);

    matplot::show();
  }

  return 0;
}
