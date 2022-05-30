#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

namespace ransac_lib {

class PlaneEstimator {
private:
  using Plane = Eigen::Hyperplane<double, 3>;
  using Vector = Eigen::Vector3d;

public:
  PlaneEstimator(const Eigen::Matrix3Xd &data);
  ~PlaneEstimator() = default;

  inline int min_sample_size() const { return 3; }

  inline int non_minimal_sample_size() const { return 6; }

  inline int num_data() const { return num_data_; }

  int MinimalSolver(const std::vector<int> &sample,
                    std::vector<Plane> *planes) const;

  int NonMinimalSolver(const std::vector<int> &sample, Plane *plane) const;

  double EvaluateModelOnPoint(const Plane &plane, int i) const;

  inline void LeastSquares(const std::vector<int> &sample, Plane *plane) const {
    NonMinimalSolver(sample, plane);
  }

private:
  Eigen::Matrix3Xd data_;
  int num_data_;
};

} // namespace ransac_lib
