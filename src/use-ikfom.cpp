#include "use-ikfom.hpp"
#include <cmath>

namespace ikfom_util
{
/************ process_noise_cov ************************************************/
MTK::get_cov<process_noise_ikfom>::type process_noise_cov()
{
    MTK::get_cov<process_noise_ikfom>::type cov =
        MTK::get_cov<process_noise_ikfom>::type::Zero();

    MTK::setDiagonal<process_noise_ikfom, vect3, 0 >(cov,&process_noise_ikfom::ng ,0.0001);
    MTK::setDiagonal<process_noise_ikfom, vect3, 3 >(cov,&process_noise_ikfom::na ,0.0001);
    MTK::setDiagonal<process_noise_ikfom, vect3, 6 >(cov,&process_noise_ikfom::nbg,0.00001);
    MTK::setDiagonal<process_noise_ikfom, vect3, 9 >(cov,&process_noise_ikfom::nba ,0.00001);
    MTK::setDiagonal<process_noise_ikfom, vect3, 12>(cov,&process_noise_ikfom::nc ,0.001);
    return cov;
}

/************ get_f ************************************************************/
Eigen::Matrix<double, 21, 1>
get_f(state_ikfom &s, const input_ikfom &in)
{
    Eigen::Matrix<double, 21, 1> res = Eigen::Matrix<double, 21, 1>::Zero();

    vect3 omega; in.gyro.boxminus(omega, s.bg);
    vect3 a_inertial = s.rot * (in.acc - s.ba);

    for(int i = 0; i < 3; ++i){
        res(i)      = omega[i];
        res(i + 3)  = s.vel[i];
        res(i + 6)  = a_inertial[i] + s.grav[i];
    }
    return res;
}

/************ df_dx ************************************************************/
Eigen::Matrix<double, 21, 20>
df_dx(state_ikfom &s, const input_ikfom &in)
{
    Eigen::Matrix<double, 21, 20> cov = Eigen::Matrix<double, 21, 20>::Zero();

    vect3 acc_;  in.acc.boxminus(acc_, s.ba);
    vect3 omega; in.gyro.boxminus(omega, s.bg);

    cov.template block<3,3>( 6, 0) = -s.rot.toRotationMatrix() * MTK::hat(acc_);
    cov.template block<3,3>(15, 0) = -s.rot.toRotationMatrix();
    cov.template block<3,3>( 3, 6) =  Eigen::Matrix3d::Identity();
    cov.template block<3,3>( 0, 9) = -Eigen::Matrix3d::Identity();
    cov.template block<3,3>( 6,12) = -s.rot.toRotationMatrix();

    Eigen::Matrix<state_ikfom::scalar,2,1> vec = Eigen::Matrix<state_ikfom::scalar,2,1>::Zero();
    Eigen::Matrix<state_ikfom::scalar,3,2> grav_matrix;
    s.S2_Mx(grav_matrix, vec, 21);
    cov.template block<3,2>(6,18) = grav_matrix;
    return cov;
}

/************ df_dw ************************************************************/
Eigen::Matrix<double, 21, 15>
df_dw(state_ikfom &s, const input_ikfom &in,
      const Eigen::Matrix3d &forward_kinematics_rotation)
{
    Eigen::Matrix<double, 21, 15> cov = Eigen::Matrix<double, 21, 15>::Zero();

    cov.template block<3,3>( 0, 0) =  Eigen::Matrix3d::Identity();
    cov.template block<3,3>( 6, 3) = -s.rot.toRotationMatrix();
    cov.template block<3,3>( 9, 6) =  Eigen::Matrix3d::Identity();
    cov.template block<3,3>(12, 9) =  Eigen::Matrix3d::Identity();
    cov.template block<3,3>(15,12) = -s.rot.toRotationMatrix()*forward_kinematics_rotation;
    return cov;
}

/************ SO3ToEuler *******************************************************/
vect3 SO3ToEuler(const SO3 &orient)
{
    Eigen::Vector3d ang;
    Eigen::Vector4d q = orient.coeffs().transpose();  // (x, y, z, w)

    double sqw = q[3]*q[3], sqx = q[0]*q[0], sqy = q[1]*q[1], sqz = q[2]*q[2];
    double unit = sqx + sqy + sqz + sqw;
    double test = q[3]*q[1] - q[2]*q[0];

    if (test >  0.49999*unit)
        ang << 2 * std::atan2(q[0], q[3]),  M_PI/2, 0;
    else if (test < -0.49999*unit)
        ang <<-2 * std::atan2(q[0], q[3]), -M_PI/2, 0;
    else
        ang <<
            std::atan2(2*q[0]*q[3] + 2*q[1]*q[2], -sqx - sqy + sqz + sqw),
            std::asin (2*test/unit),
            std::atan2(2*q[2]*q[3] + 2*q[1]*q[0],  sqx - sqy - sqz + sqw);

    ang *= 57.3;               // rad -> deg
    return vect3(ang.data(), 3);
}

} // namespace ikfom_util
