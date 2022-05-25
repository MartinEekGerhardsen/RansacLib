cmake_minimum_required(VERSION 3.0)

project(RansacLib VERSION 2.0.0)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

if(NOT TARGET Eigen3::Eigen)
  find_package(Eigen3 3.3.0 REQUIRED)
endif()

add_library(RansacLib INTERFACE)
add_library(RansacLib::RansacLib ALIAS RansacLib)

target_include_directories(
  RansacLib INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(TARGET Eigen3::Eigen)
  target_link_libraries(RansacLib INTERFACE Eigen3::Eigen)
  set(Eigen3_DEPENDENCY "find_dependency (Eigen3 ${Eigen3_VERSION})")
else()
  target_include_directories(RansacLib SYSTEM INTERFACE ${EIGEN3_INCLUDE_DIR})
endif()

set_target_properties(
  RansacLib
  PROPERTIES CXX_STANDARD 11
             CXX_STANDARD_REQUIRED YES
             CXX_EXTENSIONS NO)

option(RANSACLIB_INSTALL "Install RansacLib. " OFF)
if(RANSACLIB_INSTALL)

  install(
    TARGETS RansacLib
    EXPORT ${PROJECT_NAME}_Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

  write_basic_package_version_file(
    "RansacLibConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

  configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/RansacLibConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/RansacLibConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/RansacLib/cmake)

  install(
    EXPORT RansacLib_Targets
    FILE RansacLibTargets.cmake
    NAMESPACE RansacLib::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/RansacLib/cmake)

  install(FILES "${PROJECT_BINARY_DIR}/RansacLibConfig.cmake"
                "${PROJECT_BINARY_DIR}/RansacLibConfigVersion.cmake"
          DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/RansacLib/cmake)

  install(DIRECTORY ${PROJECT_SOURCE_DIR}/include
          DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

endif()

option(RANSACLIB_BUILD_EXAMPLES "Build examples. " OFF)
if(RANSACLIB_BUILD_EXAMPLES)

  add_subdirectory(examples)

endif()
