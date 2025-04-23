#pragma once
#include "IMU_Processing.hpp"

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ikd-Tree/ikd_Tree.h>
#include <esekfom/esekfom.hpp>
#include <random>
#include <cmath>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <algorithm>
#include <fstream>
#include <csignal>

#include <ros/ros.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Vector3.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>

#include "common_lib.h"
#include "so3_math.h"


struct Pose2D {
  double x = 0.0, y = 0.0, yaw = 0.0;
};


struct Particle {
  Pose2D pose;   // PF‐tracked (x,y,yaw)
  esekfom::esekf<state_ikfom, 15, input_ikfom> kf;     // EKF for the rest of the state
  pcl::PointCloud<PointType>::Ptr map_cloud;
  KD_TREE<PointType> ikdtree;
  double weight = 1.0;

  Particle() = default; 
};


class RBPFSLAM {
public:
    RBPFSLAM(int num_particles);

    void setExtrinsics(M3D Lidar_R, V3D Lidar_T);

    void imuPredict(const MeasureGroup &meas, const Eigen::Matrix3d &R_base_foot);
    void lidarUpdate(const MeasureGroup &meas, pcl::PointCloud<PointType>::Ptr down);
    const Particle& best() const;
    void init_dyn(esekfom::esekf<state_ikfom, 15, input_ikfom> kf);
    // void update_ikdtree(KD_TREE<PointType> ikdtree);

private:
    double solvetime = 0;
    int num_particles_;
    std::vector<Particle> particles_;

    imu_proc::ImuProcess imu_proc_;
    bool init_done_;
    int init_count_;

    M3D Lidar_R_wrt_IMU;
    V3D Lidar_T_wrt_IMU;

    std::mt19937 rng_{std::random_device{}()};
    std::uniform_real_distribution<> uni_{0.0, 1.0};


    void normalizeWeights();
    double effectiveSampleSize() const;
    void resample();
    double computePointPlaneResidual(
        const pcl::PointCloud<PointType>::Ptr &scan,
        const pcl::PointCloud<PointType>::Ptr &map_cloud,
        KD_TREE<PointType> &tree,
        esekfom::esekf<state_ikfom, 15,input_ikfom> &kf,
        Eigen::MatrixXd &H,
        Eigen::VectorXd &h);
};