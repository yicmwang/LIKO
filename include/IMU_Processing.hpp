#pragma once

#include <Eigen/Eigen>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include <nav_msgs/Odometry.h>

#include <deque>
#include <vector>
#include <fstream>

#include "common_lib.h"
#include "so3_math.h"
#include "use-ikfom.hpp"

#define MAX_INI_COUNT 10

namespace imu_proc {


bool time_list(PointType &x, PointType &y);


class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ImuProcess();
  ~ImuProcess();

  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);


  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);


  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  void set_contact_cov(const V3D &k_c);

  void Process(const MeasureGroup &meas,
               esekfom::esekf<state_ikfom, 15, input_ikfom> &kf_state,
               PointCloudXYZI::Ptr pcl_un,
               Eigen::Matrix3d     R_base_foot);
  Eigen::Matrix<double, 15, 15> Q;
  V3D cov_acc, cov_gyr, cov_acc_scale, cov_gyr_scale;
  V3D cov_bias_gyr, cov_bias_acc, cov_contact;
  double first_lidar_time;
  bool   imu_inited;



  void UndistortPcl(const MeasureGroup &meas,
                    esekfom::esekf<state_ikfom, 15, input_ikfom> &kf_state,
                    PointCloudXYZI &pcl_out,
                    Eigen::Matrix3d R_base_foot);

  void IMU_init(const MeasureGroup &meas,
                esekfom::esekf<state_ikfom, 15, input_ikfom> &kf_state,
                int &N);

  void PredictImuState(const MeasureGroup &meas,
                       esekfom::esekf<state_ikfom, 15, input_ikfom> &kf_state,
                       int num_ped_imu_meas,
                       Eigen::Matrix3d & R_base_foot);

private:
  PointCloudXYZI::Ptr  cur_pcl_un_;
  sensor_msgs::ImuConstPtr last_imu_;
  std::deque<sensor_msgs::ImuConstPtr> v_imu_;
  std::vector<Pose6D> IMUpose;
  std::vector<Pose6D> IMUposeWitOffsetT;
  std::vector<M3D>    v_rot_pcl_;
  M3D Lidar_R_wrt_IMU;
  V3D Lidar_T_wrt_IMU;
  V3D mean_gyr, angvel_last, acc_s_last;
  double start_timestamp_;
  double last_lidar_end_time;
  V3D mean_acc;
  ofstream fout_imu;
  std::deque<Pose6DTemp> IMUposeTemp;
  Pose6DTemp IMUposeTempLast;

  input_ikfom in_;
  state_ikfom imu_state_;

  int  init_iter_num;
  bool b_first_frame_;
  bool imu_need_init_;
};

} // namespace imu_proc