#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

namespace ransac_lib {

template <class Scalar = double, int Dim = 3> class PlaneEstimator {
private:
  using Plane = Eigen::Hyperplane<Scalar, Dim>;
  using Vector = Eigen::Matrix<Scalar, Dim, 1>;
  using Matrix = Eigen::Matrix<Scalar, Dim, Dim>;
  using MatrixData = Eigen::Matrix<Scalar, Dim, Eigen::Dynamic>;
  using PlanarVectors = Eigen::Matrix<Scalar, Dim - 1, Dim>;

public:
  PlaneEstimator(const MatrixData &data)
      : data_{data}, num_data_{data.cols()} {}
  ~PlaneEstimator() = default;

  inline int min_sample_size() const { return Dim; }

  inline int non_minimal_sample_size() const { return 2 * Dim; }

  inline int num_data() const { return num_data_; }

  int MinimalSolver(const std::vector<int> &sample,
                    std::vector<Plane> *planes) const {
    planes->clear();
    if (sample.size() < min_sample_size()) {
      return 0;
    }

    Vector base = data_.col(sample[0]);
    PlanarVectors planarVectors;
    for (auto it{sample.begin() + 1}; it != sample.end(); ++it) {
      planarVectors.row(*it).array() = (data_.col(sample[*it]) - base).array();
    }

    Vector normal;
    for (int dim{0}; dim < Dim; ++dim) {
      Matrix det;
      det << planarVectors.template leftCols<dim>(),
          planarVectors.template rightCols<Dim - dim>();
      normal(dim) = det.determinant();
    }
  }

  int NonMinimalSolver_ldlt(const std::vector<int> &sample, Plane *plane) const;
  int MinimalSolver_cov_eigen_cross(const std::vector<int> &sample,
                                    Plane *plane) const;
  int MinimalSolver_cov_eigen_min(const std::vector<int> &sample,
                                  Plane *plane) const;

  int NonMinimalSolver(const std::vector<int> &sample, Plane *plane) const;

  double EvaluateModelOnPoint(const Plane &plane, int i) const;

  inline void LeastSquares(const std::vector<int> &sample, Plane *plane) const {
    NonMinimalSolver(sample, plane);
  }

private:
  MatrixData data_;
  Eigen::Index num_data_;
};

} // namespace ransac_lib
