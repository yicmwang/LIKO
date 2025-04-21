#include "IMU_Processing.hpp"

#include <ros/ros.h>
#include <pcl/common/io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>

#include <omp.h> 

namespace imu_proc {
bool time_list(PointType &x, PointType &y)
{
    return (x.curvature < y.curvature);
}

ImuProcess::ImuProcess()
    : b_first_frame_(true),
      imu_need_init_(true),
      start_timestamp_(-1),
      imu_inited(false)
{
    init_iter_num = 1;
    Q = ikfom_util::process_noise_cov();

    cov_acc       = V3D(0.1, 0.1, 0.1);
    cov_gyr       = V3D(0.1, 0.1, 0.1);
    cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
    cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);

    mean_acc      = V3D(0, 0, -1.0);
    mean_gyr      = V3D(0, 0, 0);
    angvel_last   = Zero3d;

    Lidar_T_wrt_IMU = Zero3d;
    Lidar_R_wrt_IMU = Eye3d;

    last_imu_.reset(new sensor_msgs::Imu());
}

ImuProcess::~ImuProcess() = default;

void ImuProcess::Reset()
{
    mean_acc      = V3D(0, 0, -1.0);
    mean_gyr      = V3D(0, 0, 0);
    angvel_last   = Zero3d;
    imu_need_init_    = true;
    start_timestamp_  = -1;
    init_iter_num     = 1;
    v_imu_.clear();
    last_imu_.reset(new sensor_msgs::Imu());
    cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
    Lidar_T_wrt_IMU = T.block<3,1>(0,3);
    Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

void ImuProcess::set_contact_cov(const V3D &k_c)
{
  cov_contact = k_c;
}

void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 15, input_ikfom> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance
   ** 2. normalize the acceleration measurenments to unit gravity **/
  
  V3D cur_acc, cur_gyr;
  
  if (b_first_frame_)
  {
    Reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    first_lidar_time = meas.lidar_beg_time;
  }

  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N);

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N ++;
  }
  state_ikfom init_state = kf_state.get_x();
  init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);
  
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state.bg  = mean_gyr;
  // init_state.offset_T_L_I = Lidar_T_wrt_IMU;
  // init_state.offset_R_L_I = Lidar_R_wrt_IMU;
  kf_state.change_x(init_state);

  esekfom::esekf<state_ikfom, 15, input_ikfom>::cov init_P = kf_state.get_P();
  init_P.setIdentity();
  // init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  // init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  // init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  // init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  // init_P(21,21) = init_P(22,22) = 0.00001; 

  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.0001;
  init_P(12,12) = init_P(13,13) = init_P(14,14) = 0.001;
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.01;
  init_P(18,18) = init_P(19,19) = 0.00001;
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();
}

void ImuProcess::PredictImuState(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 15, input_ikfom> &kf_state, int num_ped_imu_meas, Eigen::Matrix3d & R_base_foot)
{
  if(imu_need_init_)
  {
    return;
  }
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  imu_state_ = kf_state.get_x();
  if(num_ped_imu_meas == 0)
  {
    IMUpose.clear();
    IMUposeTemp.clear();
    IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state_.vel, imu_state_.pos, imu_state_.rot.toRotationMatrix()));
  }
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;
  double dt = 0;
  for (auto it_imu = (v_imu.begin()+num_ped_imu_meas); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);
    
    if (tail->header.stamp.toSec() < last_lidar_end_time)
    {
      continue;
    }
      
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);
    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;
    if(head->header.stamp.toSec() < last_lidar_end_time)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    in_.acc = acc_avr;
    in_.gyro = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    Q.block<3, 3>(12, 12).diagonal() = cov_contact;
    kf_state.predict(dt, Q, in_, R_base_foot); // 预测。
    /* save the poses at each IMU measurements */
    imu_state_ = kf_state.get_x();
    angvel_last = angvel_avr - imu_state_.bg;
    acc_s_last  = imu_state_.rot * (acc_avr - imu_state_.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state_.grav[i];
    }
    IMUposeTempLast.acc_s_last_temp = acc_s_last;
    IMUposeTempLast.angvel_last_temp = angvel_last;
    IMUposeTempLast.imu_vel_temp = imu_state_.vel;
    IMUposeTempLast.imu_pos_temp = imu_state_.pos;
    IMUposeTempLast.imu_rot_temp = imu_state_.rot.toRotationMatrix();
    IMUposeTempLast.imu_time_temp = tail->header.stamp.toSec();
    IMUposeTemp.push_back(IMUposeTempLast);
  }
}

void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 15, input_ikfom> &kf_state, PointCloudXYZI &pcl_out, Eigen::Matrix3d R_base_foot)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double &pcl_beg_time = meas.lidar_beg_time;
  const double &pcl_end_time = meas.lidar_end_time;
  
  /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), imu_proc::time_list);
  // /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;
  for (auto it_imu = IMUposeTemp.begin(); it_imu < (IMUposeTemp.end()); it_imu++)
  {
    auto &&head = *(it_imu);
    double &&offs_t = head.imu_time_temp - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, head.acc_s_last_temp, head.angvel_last_temp, head.imu_vel_temp, head.imu_pos_temp, head.imu_rot_temp));

  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in_, R_base_foot); // 预测。
  
  imu_state_ = kf_state.get_x();
  last_imu_ = meas.imu.back();
  last_lidar_end_time = pcl_end_time;

  /*** undistort each lidar point (backward propagation) ***/
  if (pcl_out.points.begin() == pcl_out.points.end()) return;
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot);
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel);
    pos_imu<<VEC_FROM_ARRAY(head->pos);
    acc_imu<<VEC_FROM_ARRAY(tail->acc);
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr);

    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      dt = it_pcl->curvature / double(1000) - head->offset_time;
      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in_ global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state_.pos);
      V3D P_compensate = Lidar_R_wrt_IMU.conjugate() * (imu_state_.rot.conjugate() * (R_i * (Lidar_R_wrt_IMU * P_i + Lidar_T_wrt_IMU) + T_ei) - Lidar_T_wrt_IMU);// not accurate!
      
      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }

}

void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 15, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_, Eigen::Matrix3d R_base_foot)
{
  double t1,t2,t3;
  t1 = omp_get_wtime();

  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.lidar != nullptr);

  if (imu_need_init_)
  {
    /// The very first lidar frame
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    imu_state_ = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT)
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2);
      imu_need_init_ = false;

      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
      // ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }
    imu_inited = true;
    return;
  }

  UndistortPcl(meas, kf_state, *cur_pcl_un_, R_base_foot);

  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
} // namespace imu_proc
