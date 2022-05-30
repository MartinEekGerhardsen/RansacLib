
#include <Eigen/Eigenvalues>

#include "plane_estimator.hpp"

namespace ransac_lib {

PlaneEstimator::PlaneEstimator(const Eigen::Matrix3Xd &data) {
  data_ = data;
  num_data_ = static_cast<int>(data.cols());
}

int PlaneEstimator::MinimalSolver(const std::vector<int> &sample,
                                  std::vector<Plane> *planes) const {
  planes->clear();
  if (sample.size() < min_sample_size()) {
    return 0;
  }

  planes->reserve(1);
  Vector p0 = data_.col(sample[0]);
  Vector p1p0 = data_.col(sample[1]) - p0;
  Vector p2p0 = data_.col(sample[2]) - p0;

  Vector valid = p1p0.array() / p2p0.array();

  if ((valid(0) == valid(1) && (valid(1) == valid(2)))) {
    return 0;
  }

  Vector normal = p1p0.cross(p2p0);
  normal.normalize();

  planes->emplace_back(normal, p0);

  return 1;
}

int PlaneEstimator::NonMinimalSolver(const std::vector<int> &sample,
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
  Vector normal = eigensolver.eigenvectors().col(index).real();
  normal.normalize();

  *plane = Plane(normal, mean);

  return 1;
}

double PlaneEstimator::EvaluateModelOnPoint(const Plane &plane, int i) const {
  return plane.absDistance(data_.col(i));
}

} // namespace ransac_lib
