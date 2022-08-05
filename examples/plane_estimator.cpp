#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <algorithm>
#include <cmath>
#include <iostream>

#include "plane_estimator.hpp"

namespace ransac_lib {

// int PlaneEstimator::MinimalSolver(const std::vector<int> &sample,
//                                   std::vector<Plane> *planes) const {
//   planes->clear();
//   if (sample.size() < min_sample_size()) {
//     return 0;
//   }

//   planes->reserve(1);
//   Vector p0 = data_.col(sample[0]);
//   Vector p1p0 = data_.col(sample[1]) - p0;
//   Vector p2p0 = data_.col(sample[2]) - p0;

//   Vector valid = p1p0.array() / p2p0.array();

//   if ((valid(0) == valid(1) && (valid(1) == valid(2)))) {
//     return 0;
//   }

//   Vector normal = p1p0.cross(p2p0);
//   normal.normalize();

//   planes->emplace_back(normal, -normal.dot(p0));

//   for (Plane &plane : *planes)
//     plane.normalize();

//   return planes->size();
// }

/**
 * @brief
 *
 * @param sample
 * @param plane
 * @return int
 *>: - a * x / d - b * y / d - c * z / d = 1.0
 * alpha * x + beta * y + gamma * z = 1.0
 * Then: A = [[x_1, y_1, z_1]  b = [1.0
 *             ...  ...  ...        ...
 *            [x_n, y_n, z_n]]      1.0]
 *      xhi = [alpha, beta, gamma] = (A^T * A)^(-1) * A^T b
 * And n = [a, b, c] = [alpha, beta, gamma],
 * d = -1.0
 * To get valid plane, this is normalized
 */
int PlaneEstimator::NonMinimalSolver_ldlt(const std::vector<int> &sample,
                                          Plane *plane) const {
  if (sample.size() < non_minimal_sample_size()) {
    return 0;
  }

  const int kNumSamples = static_cast<int>(sample.size());

  Eigen::MatrixXd A = Eigen::MatrixXd(kNumSamples, 3);
  Eigen::VectorXd b = Eigen::VectorXd::Constant(kNumSamples, 1.0);

  for (int i{0}; i < sample.size(); ++i) {
    A.block<1, 3>(i, 0).array() = data_.block<3, 1>(0, sample[i]).array();
  }

  Eigen::Vector3d normal = (A.transpose() * A).ldlt().solve(A.transpose() * b);

  *plane = Plane(normal, -1.0);
  plane->normalize();

  return 1;
}

int PlaneEstimator::MinimalSolver_cov_eigen_cross(
    const std::vector<int> &sample, Plane *plane) const {
  if (sample.size() < non_minimal_sample_size()) {
    return 0;
  }

  const int kNumSamples = static_cast<int>(sample.size());

  Vector mean = Vector::Zero();
  for (int s : sample) {
    mean += data_.col(s);
  }

  mean /= static_cast<double>(kNumSamples);

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();

  for (int s : sample) {
    Vector centered = data_.col(s) - mean;
    covariance += centered * centered.transpose();
  }

  covariance /= static_cast<double>(kNumSamples - 1);

  Eigen::EigenSolver<Eigen::Matrix3d> eigensolver(covariance);

  if (eigensolver.info() != Eigen::Success) {
    return 0;
  }

  Eigen::Array3i indices = Eigen::Array3i::LinSpaced(0, 3);
  const Eigen::Vector3d eigenvalues = eigensolver.eigenvalues().real();
  const Eigen::Matrix3d eigenvectors = eigensolver.eigenvectors().real();

  std::sort(indices.data(), indices.data() + indices.size(),
            [&](int i, int j) { return eigenvalues(i) > eigenvalues(j); });

  const Vector planar0 = eigenvectors.col(indices(0));
  const Vector planar1 = eigenvectors.col(indices(1));
  const Vector normal = planar0.cross(planar1);

  *plane = Plane(normal, mean);
  plane->normalize();

  return 1;
}

int PlaneEstimator::MinimalSolver_cov_eigen_min(const std::vector<int> &sample,
                                                Plane *plane) const {
  if (sample.size() < non_minimal_sample_size()) {
    return 0;
  }

  const int kNumSamples = static_cast<int>(sample.size());

  Vector mean = Vector::Zero();
  for (int s : sample) {
    mean += data_.col(s);
  }

  mean /= static_cast<double>(kNumSamples);

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();

  for (int s : sample) {
    Vector centered = data_.col(s) - mean;
    covariance += centered * centered.transpose();
  }

  covariance /= static_cast<double>(kNumSamples - 1);

  Eigen::EigenSolver<Eigen::Matrix3d> eigensolver(covariance);

  if (eigensolver.info() != Eigen::Success) {
    return 0;
  }

  std::size_t index{0};
  eigensolver.eigenvalues().real().minCoeff(&index);
  const Vector normal = eigensolver.eigenvectors().col(index).real();

  *plane = Plane(normal, mean);
  plane->normalize();

  return 1;
}

int PlaneEstimator::NonMinimalSolver(const std::vector<int> &sample,
                                     Plane *plane) const {

  return NonMinimalSolver_ldlt(sample, plane);
}

double PlaneEstimator::EvaluateModelOnPoint(const Plane &plane, int i) const {
  // bool stop = ((plane.normal().x() < -0.5) && (plane.normal().z() < -0.5) &&
  //              ((plane.offset() < -6) || (plane.offset() > 6)));

  // Vector point = data_.col(i);
  // double distance = plane.signedDistance(point);

  // double squaredError = std::pow(distance, 2);

  return std::pow(plane.absDistance(data_.col(i)), 2);
}

} // namespace ransac_lib
