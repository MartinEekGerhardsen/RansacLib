find_package(Ceres QUIET)
find_package(PoseLib QUIET)

add_definitions(-march=native)
if(MSVC)
  add_definitions(-D_USE_MATH_DEFINES)
endif()

if(PoseLib AND Ceres)

  include_directories(${EIGEN3_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS}
                      ${POSELIB_INCLUDE_DIRS})

  add_executable(
    camera_pose_estimation
    camera_pose_estimation.cpp calibrated_absolute_pose_estimator.cpp
    calibrated_absolute_pose_estimator.hpp)
  target_link_libraries(camera_pose_estimation ${POSELIB_LIBRARIES}
                        ${CERES_LIBRARIES} RansacLib)

  add_executable(
    localization localization.cpp calibrated_absolute_pose_estimator.cpp
                 calibrated_absolute_pose_estimator.hpp)
  target_link_libraries(localization ${POSELIB_LIBRARIES} ${CERES_LIBRARIES}
                        RansacLib)

  add_executable(
    localization_with_gt
    localization_with_gt.cpp calibrated_absolute_pose_estimator.cpp
    calibrated_absolute_pose_estimator.hpp)
  target_link_libraries(localization_with_gt ${POSELIB_LIBRARIES}
                        ${CERES_LIBRARIES} RansacLib)

endif()

add_executable(line_estimation line_estimation.cpp line_estimator.cpp
                               line_estimator.hpp)
target_link_libraries(line_estimation RansacLib)

add_executable(
  hybrid_line_estimation hybrid_line_estimation.cpp hybrid_line_estimator.cpp
                         hybrid_line_estimator.hpp)
target_link_libraries(hybrid_line_estimation RansacLib)

add_executable(plane_estimation plane_estimation.cpp plane_estimator.cpp
                                plane_estimator.hpp)
target_link_libraries(plane_estimation RansacLib)
