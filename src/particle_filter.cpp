#include "particle_filter.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <ikd-Tree/ikd_Tree.h>

RBPFSLAM::RBPFSLAM(int num_particles)
  : num_particles_(num_particles),
    particles_(num_particles),
    init_done_(false),
    init_count_(1)
{}

void RBPFSLAM::setExtrinsics(M3D Lidar_R, V3D Lidar_T){
    Lidar_R_wrt_IMU = Lidar_R;
    Lidar_T_wrt_IMU = Lidar_T;
}

void RBPFSLAM::init_dyn(esekfom::esekf<state_ikfom, 15, input_ikfom> kf){
    particles_[0].kf = kf;
    for (int i = 1; i < num_particles_; ++i)
            particles_[i].kf = particles_[0].kf;
}

void RBPFSLAM::imuPredict(const MeasureGroup &meas, const Eigen::Matrix3d &R_base_foot) {


    // 1) 首次呼叫時用 IMU_init 初始化 EKF，並且把它複製給所有粒子
    if (!init_done_) {
        imu_proc_.IMU_init(meas, particles_[0].kf, init_count_);
        for (int i = 1; i < num_particles_; ++i)
            particles_[i].kf = particles_[0].kf;
        init_done_ = true;
        return;  // 一定要在這裡 return，後面不要跑 PF update
    }

    // 2) 拿整段 IMU 資料算平面運動量
    double dt = meas.imu.back()->header.stamp.toSec()
              - meas.imu.front()->header.stamp.toSec();
    const auto &imu = *meas.imu.back();
    double wz   = imu.angular_velocity.z;
    double ax_b = imu.linear_acceleration.x;
    double ay_b = imu.linear_acceleration.y;


    // 3) 用 PF 做平面 (x,y,yaw) 抽樣，再把新 pose 餵回各自的 EKF
    for (auto &p : particles_) {
        // 3‑1) 粗平面運動模型
        double c  = std::cos(p.pose.yaw),
               s  = std::sin(p.pose.yaw);
        double vx = ax_b * dt + std::normal_distribution<>(0,0.05)(rng_);
        double vy = ay_b * dt + std::normal_distribution<>(0,0.05)(rng_);
        double wz_s = wz + std::normal_distribution<>(0,0.02)(rng_);

        p.pose.x   +=  c*vx - s*vy;
        p.pose.y   +=  s*vx + c*vy;
        p.pose.yaw +=  wz_s * dt;


        // 3‑2) 將 PF 得到的 (x,y,yaw) “灌回” EKF state，EKF 只繼續預測剩下 15 維
        auto st = p.kf.get_x();
        // overwrite yaw only (保持 roll/pitch 由 EKF 積分)
        Eigen::Matrix3d Rwb_old = st.rot.toRotationMatrix();
        Eigen::Vector3d zyx = Rwb_old.eulerAngles(2,1,0);
        double yaw_old   = zyx[0];
        double pitch_old = zyx[1];
        double roll_old  = zyx[2];

        Eigen::Matrix3d Rwb_new =
            (Eigen::AngleAxisd(p.pose.yaw, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(pitch_old, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(roll_old, Eigen::Vector3d::UnitX()))
            .toRotationMatrix();
        // 下面這行只更新 Z 軸的方向向量，保持 roll/pitch 不變
        st.rot = Eigen::Quaterniond(Rwb_new);

        st.pos.x() = p.pose.x;
        st.pos.y() = p.pose.y;
        p.kf.change_x(st);  

        // 最後呼叫 EKF 的 predict 來更新剩下的 15 維
        imu_proc_.PredictImuState(meas, p.kf, 0, const_cast<Eigen::Matrix3d&>(R_base_foot));
    }
}

// void RBPFSLAM::update_ikdtree(KD_TREE<PointType> ikdtree){
//     for (auto &p : particles_) {
//         p.ikdtree = ikdtree;
//     }
// }

// Lidar Update
void RBPFSLAM::lidarUpdate(const MeasureGroup &meas,
                           pcl::PointCloud<PointType>::Ptr down)
{
    ROS_INFO("start");
    constexpr double sigma2 = 0.02;  // plane‐residual variance


    if (!down || down->empty()) {
        ROS_WARN("Downsampled cloud is null or empty. Skip lidarUpdate.");
        return;
    }

    Eigen::MatrixXd H; Eigen::VectorXd h;

    for (auto &p : particles_) {
        // (1) slam pose → EKF state
        auto s = p.kf.get_x();
        s.pos.x() = p.pose.x;
        s.pos.y() = p.pose.y;
        s.rot     = Eigen::Quaterniond{
                      Eigen::AngleAxisd(p.pose.yaw,
                                        Eigen::Vector3d::UnitZ())
                    };
        p.kf.change_x(s);
        
        pcl::PointCloud<PointType>::Ptr world_scan(new pcl::PointCloud<PointType>);
        world_scan->reserve(down->size());

        Eigen::Matrix3d Rwb = s.rot.toRotationMatrix();
        Eigen::Vector3d pwb = s.pos;

        for (const auto &pt : *down) {
            Eigen::Vector3d pl(pt.x,pt.y,pt.z);
            Eigen::Vector3d pb = Lidar_R_wrt_IMU * pl + Lidar_T_wrt_IMU;
            Eigen::Vector3d pw = Rwb * pb + pwb;
            PointType w; w.x = pw.x(); w.y = pw.y(); w.z = pw.z();
            world_scan->push_back(w);
        }

        ROS_INFO("what ever");
        if (!p.map_cloud) {
            p.map_cloud = world_scan;
            p.ikdtree.Build(p.map_cloud->points);
        }
        else {
            p.map_cloud->points.insert(p.map_cloud->points.end(),
                                       world_scan->points.begin(),
                                       world_scan->points.end());
            ROS_INFO("add_points, world scan size %d", world_scan->points.size());
            p.ikdtree.Add_Points(world_scan->points, /*rebuild=*/false);
        }


        // auto down = downsample(pcl);

        // (2) compute point‐to‐plane residual + Jacobian
        ROS_INFO("computePointPlaneResidual");
        double res = computePointPlaneResidual(
                       down, p.map_cloud, p.ikdtree,
                       p.kf, H, h);

        // (3) zero out the PF (x,y,yaw) cols → only update the other 15 dims
        //H.block(0, 0, H.rows(), 6).setZero();
        ROS_INFO("update_iterated_dyn_share_modified");
        if (H.rows() > 0) {
            // H.block(0,  2, H.rows(), 1).setZero();
            // H.block(0,  6, H.rows(), 2).setZero();
            // Eigen::MatrixXd R = Eigen::MatrixXd::Identity(h.size(), h.size()) * sigma2;
            p.kf.update_iterated_dyn_share_modified(sigma2, solvetime, 4);
        }

        // (5) re‐weight PF
        p.weight *= std::exp(-0.5 * res / sigma2);
        ROS_INFO("done");
    }
    ROS_INFO("normalizeWeights");
    normalizeWeights();
    if (effectiveSampleSize() < 0.5 * num_particles_){
        resample();
    }
        
}



const Particle& RBPFSLAM::best() const {
    return *std::max_element(
        particles_.begin(), particles_.end(),
        [](auto &a, auto &b){ return a.weight < b.weight; }
    );
}

void RBPFSLAM::normalizeWeights() {
    double sum = 0;
    for (auto &p : particles_) sum += p.weight;
    for (auto &p : particles_) p.weight /= sum;
}

double RBPFSLAM::effectiveSampleSize() const {
    double sumsq = 0;
    for (auto &p : particles_) sumsq += p.weight * p.weight;
    return 1.0 / sumsq;
}

void RBPFSLAM::resample()
{
    std::vector<Particle> newset(num_particles_);

    // 1) 系統化重採樣
    std::vector<double> cdf(num_particles_);
    cdf[0] = particles_[0].weight;
    for (int i = 1; i < num_particles_; ++i) cdf[i] = cdf[i-1] + particles_[i].weight;
    double u0 = uni_(rng_) / num_particles_;
    int idx = 0;

    for (int m = 0; m < num_particles_; ++m) {
        double u = u0 + m * (1.0 / num_particles_);
        while (u > cdf[idx]) ++idx;

        // 2) 淺拷貝 ‑‑ map雲＆KDTree 指標共用
        newset[m] = particles_[idx];
        newset[m].weight = 1.0 / num_particles_;
    }
    particles_.swap(newset);
}



double RBPFSLAM::computePointPlaneResidual(
    const pcl::PointCloud<PointType>::Ptr &scan,
    const pcl::PointCloud<PointType>::Ptr & map_cloud,
    KD_TREE<PointType> &tree,
    esekfom::esekf<state_ikfom,15,input_ikfom> &kf,
    Eigen::MatrixXd &H,
    Eigen::VectorXd &h)
{

  if (!scan || scan->empty()) {
    ROS_WARN("Scan is null or empty in computePointPlaneResidual.");
    return 0.0;

  }
    ROS_INFO(" number of measurements");
    
    
  // number of measurements
  int N = scan->points.size();
  H.setZero(N, 15);    // 15‑state ESKF: [rot(3), vel(3), pos(3), bg(3), ba(3)]
  h.resize(N);

  // current state
  auto s = kf.get_x();
  Eigen::Matrix3d Rwb = s.rot.toRotationMatrix();
  Eigen::Vector3d pwb = s.pos;

  double sum_sq = 0.0;
  const int K = NUM_MATCH_POINTS;
  auto pts_near = KD_TREE<PointType>::PointVector(K);
  // std::vector<PointType> pts_near(K);
  std::vector<float> dists(K);
    ROS_INFO("for");
  ROS_INFO("pts_near size = %d", pts_near.size());
  for(int i=0;i<N;++i){
    
    // ROS_INFO("esti_plane iter %d", i);
    const auto &pt = scan->points[i];

    // ---- KNN with world coordinates ----
    Eigen::Vector3d p_lidar(pt.x,pt.y,pt.z);
    Eigen::Vector3d p_body  = Lidar_R_wrt_IMU * p_lidar + Lidar_T_wrt_IMU;
    Eigen::Vector3d p_world = Rwb * p_body + pwb;
    PointType query;
    query.x = static_cast<float>(p_world.x());
    query.y = static_cast<float>(p_world.y());
    query.z = static_cast<float>(p_world.z());
    // ROS_INFO("Nearest_Search, %d", pts_near.size());
    tree.Nearest_Search(query, K, pts_near, dists, 5.0f);


    if (pts_near.size() < NUM_MATCH_POINTS) {
        ROS_WARN("esti_plane: not enough neighbors! got %zu, require %d. iteration number %d",
        pts_near.size(), NUM_MATCH_POINTS, i);;
        // for (int j = pts_near.size(); j < NUM_MATCH_POINTS; ++j)
        //     pts_near.push_back(pts_near.back());
    }
    // std::cout << pts_near.size();
    // plane fit
    Eigen::Vector4d abcd;
    // ROS_INFO("esti_plane");
    if(!esti_plane(abcd, pts_near, 0.1f)){
        // ROS_INFO("esti_plane if statement");
        H.row(i).setZero(); 
        h(i)=0; 
        //  ROS_INFO("esti_plane if statement done");
        continue; 
        }

    Eigen::Vector3d normal(abcd[0], abcd[1], abcd[2]);

    // residual
    double pd = normal.dot(p_world) + abcd(3);
    h(i) = -pd;
    sum_sq += pd*pd;
    // ROS_INFO("Jacobian");

    // Jacobian wrt position
    H(i,6) = normal.x();
    H(i,7) = normal.y();
    H(i,8) = normal.z();

    // Jacobian wrt small rotation
    // ROS_INFO("skew");
    Eigen::Matrix3d skew;
    skew <<           0, -p_body.z(),  p_body.y(),
             p_body.z(),          0, -p_body.x(),
            -p_body.y(),  p_body.x(),           0;
    Eigen::RowVector3d J_rot = normal.transpose() * (Rwb * skew);
    H(i,0)=J_rot.x(); H(i,1)=J_rot.y(); H(i,2)=J_rot.z();
    }
  return sum_sq;
}



// void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 15, input_ikfom> &kf_state, PointCloudXYZI &pcl_out, Eigen::Matrix3d R_base_foot)
// {
//   /*** add the imu of the last frame-tail to the of current frame-head ***/
//   ROS_INFO("v_imu");
//   auto v_imu = meas.imu;
//   ROS_INFO("push front");
//   v_imu.push_front(last_imu_);
//   const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
//   const double &imu_end_time = v_imu.back()->header.stamp.toSec();
//   const double &pcl_beg_time = meas.lidar_beg_time;
//   const double &pcl_end_time = meas.lidar_end_time;
  
//   /*** sort point clouds by offset time ***/
//   ROS_INFO("sort");
//   pcl_out = *(meas.lidar);
//   sort(pcl_out.points.begin(), pcl_out.points.end(), imu_proc::time_list);
//   // /*** forward propagation at each imu point ***/
//   V3D angvel_avr, acc_imu, vel_imu, pos_imu;
//   M3D R_imu;

//   ROS_INFO("push_back imu pose");
//   double dt = 0;
//   for (auto it_imu = IMUposeTemp.begin(); it_imu < (IMUposeTemp.end()); it_imu++)
//   {
//     auto &&head = *(it_imu);
//     double &&offs_t = head.imu_time_temp - pcl_beg_time;
//     IMUpose.push_back(set_pose6d(offs_t, head.acc_s_last_temp, head.angvel_last_temp, head.imu_vel_temp, head.imu_pos_temp, head.imu_rot_temp));

//   }
//   ROS_INFO("prediction");
//   /*** calculated the pos and attitude prediction at the frame-end ***/
//   double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
//   dt = note * (pcl_end_time - imu_end_time);
//   kf_state.predict(dt, Q, in_, R_base_foot); // 预测。
  
//   imu_state_ = kf_state.get_x();
//   last_imu_ = meas.imu.back();
//   last_lidar_end_time = pcl_end_time;

//   ROS_INFO("undistort");
//   /*** undistort each lidar point (backward propagation) ***/
//   if (pcl_out.points.begin() == pcl_out.points.end()) return;
//   auto it_pcl = pcl_out.points.end() - 1;
//   for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
//   {
//     auto head = it_kp - 1;
//     auto tail = it_kp;
//     R_imu<<MAT_FROM_ARRAY(head->rot);
//     // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
//     vel_imu<<VEC_FROM_ARRAY(head->vel);
//     pos_imu<<VEC_FROM_ARRAY(head->pos);
//     acc_imu<<VEC_FROM_ARRAY(tail->acc);
//     angvel_avr<<VEC_FROM_ARRAY(tail->gyr);

//     for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
//     {
//       dt = it_pcl->curvature / double(1000) - head->offset_time;
//       /* Transform to the 'end' frame, using only the rotation
//        * Note: Compensation direction is INVERSE of Frame's moving direction
//        * So if we want to compensate a point at timestamp-i to the frame-e
//        * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in_ global frame */
//       M3D R_i(R_imu * Exp(angvel_avr, dt));
      
//       V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
//       V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state_.pos);
//       V3D P_compensate = Lidar_R_wrt_IMU.conjugate() * (imu_state_.rot.conjugate() * (R_i * (Lidar_R_wrt_IMU * P_i + Lidar_T_wrt_IMU) + T_ei) - Lidar_T_wrt_IMU);// not accurate!
      
//       // save Undistorted points and their rotation
//       it_pcl->x = P_compensate(0);
//       it_pcl->y = P_compensate(1);
//       it_pcl->z = P_compensate(2);

//       if (it_pcl == pcl_out.points.begin()) break;
//     }
//   }

// }