#include "particle_filter.hpp"
#include <algorithm>
#include <numeric>
#include <random>


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
            (Eigen::AngleAxisd(p.pose.yaw,  Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(pitch_old,   Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(roll_old,    Eigen::Vector3d::UnitX()))
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


// Lidar Update
void RBPFSLAM::lidarUpdate(const MeasureGroup &meas,
                           pcl::PointCloud<PointType>::Ptr down)
{
    constexpr double sigma2 = 0.02;  // plane‐residual variance
    Eigen::MatrixXd H;
    Eigen::VectorXd h;

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

        // (2) compute point‐to‐plane residual + Jacobian
        double res = computePointPlaneResidual(
                       down, p.map_cloud, p.ikdtree,
                       p.kf, H, h);

        // (3) zero out the PF (x,y,yaw) cols → only update the other 15 dims
        //H.block(0, 0, H.rows(), 6).setZero();
        H.block(0, 2, H.rows(), 1).setZero();   // 清掉第 2 欄 (Δyaw)
        H.block(0, 6, H.rows(), 2).setZero();

        // (4) build a σ²·I measurement‐noise matrix and do the iterated EKF
        Eigen::MatrixXd R = Eigen::MatrixXd::Identity(h.size(), h.size()) * sigma2;
        p.kf.update_iterated_dyn(h, R);

        // (5) re‐weight PF
        p.weight *= std::exp(-0.5 * res / sigma2);
    }

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

// double RBPFSLAM::computePointPlaneResidual(
//     const pcl::PointCloud<PointType>::Ptr &scan,
//     const pcl::PointCloud<PointType>::Ptr &map_cloud,
//     ikd::KDTree<PointType>               &tree,
//     esekfom::esekf<state_ikfom,15,input_ikfom> &kf,
//     Eigen::VectorXd                       &h_x,
//     Eigen::VectorXd                       &h)
// {
//     // TODO: copy LIKO's surfJacobian & pointAssociateToMap logic here.
//     //       Populate h_x,h and accumulate squared residuals.
//     return 0.0;
// }




double RBPFSLAM::computePointPlaneResidual(
    const pcl::PointCloud<PointType>::Ptr &scan,
    const pcl::PointCloud<PointType>::Ptr & map_cloud,
    KD_TREE<PointType> &tree,
    esekfom::esekf<state_ikfom,15,input_ikfom> &kf,
    Eigen::MatrixXd &H,
    Eigen::VectorXd &h)
{
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
  typename KD_TREE<PointType>::PointVector pts_near(K);
  // std::vector<PointType> pts_near(K);
  std::vector<float> dists(K);

  for(int i=0;i<N;++i){
    const auto &pt = scan->points[i];

    // ---- KNN with world coordinates ----
    Eigen::Vector3d p_lidar(pt.x,pt.y,pt.z);
    Eigen::Vector3d p_body  = Lidar_R_wrt_IMU * p_lidar + Lidar_T_wrt_IMU;
    Eigen::Vector3d p_world = Rwb * p_body + pwb;
    PointType query;
    query.x = static_cast<float>(p_world.x());
    query.y = static_cast<float>(p_world.y());
    query.z = static_cast<float>(p_world.z());
    tree.Nearest_Search(query, K, pts_near, dists, 0.30f);

    // plane fit
    Eigen::Vector4f abcd;
    if(!esti_plane(abcd, pts_near, 0.1f)){ h(i)=0; continue; }

    Eigen::Vector3d normal(abcd[0], abcd[1], abcd[2]);

    // residual
    double pd = normal.dot(p_world) + abcd(3);
    h(i) = -pd;
    sum_sq += pd*pd;

    // Jacobian wrt position
    H(i,6) = normal.x();
    H(i,7) = normal.y();
    H(i,8) = normal.z();

    // Jacobian wrt small rotation
    Eigen::Matrix3d skew;
    skew <<           0, -p_body.z(),  p_body.y(),
             p_body.z(),          0, -p_body.x(),
            -p_body.y(),  p_body.x(),           0;
    Eigen::RowVector3d J_rot = normal.transpose() * (Rwb * skew);
    H(i,0)=J_rot.x(); H(i,1)=J_rot.y(); H(i,2)=J_rot.z();
    }

  return sum_sq;
}
