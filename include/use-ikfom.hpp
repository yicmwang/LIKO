#pragma once
#include <Eigen/Eigen>
#include <IKFoM_toolkit/esekfom/esekfom.hpp>


typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double>     SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2;
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

MTK_BUILD_MANIFOLD(state_ikfom,
((SO3,  rot))
((vect3,pos))
((vect3,vel))
((vect3,bg))
((vect3,ba))
((vect3,pc))
((S2,  grav))
);

MTK_BUILD_MANIFOLD(input_ikfom,
((vect3, acc))
((vect3, gyro))
);

MTK_BUILD_MANIFOLD(process_noise_ikfom,
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
((vect3, nc))
);

namespace ikfom_util
{

    MTK::get_cov<process_noise_ikfom>::type
    process_noise_cov();


    Eigen::Matrix<double, 21, 1>
    get_f(state_ikfom &s, const input_ikfom &in);


    Eigen::Matrix<double, 21, 20>
    df_dx(state_ikfom &s, const input_ikfom &in);

    Eigen::Matrix<double, 21, 15>
    df_dw(state_ikfom &s, const input_ikfom &in,
          const Eigen::Matrix3d &forward_kinematics_rotation);

    /* SO3 → Euler (roll, pitch, yaw) [deg] */
    vect3 SO3ToEuler(const SO3 &orient);
} // namespace ikfom_util
