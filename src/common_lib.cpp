#include "common_lib.h"

M3D Eye3d = M3D::Identity();
M3F Eye3f = M3F::Identity();
V3D Zero3d(0,0,0);
V3F Zero3f(0,0,0);

float calc_dist(PointType p1, PointType p2)
{
    return (p1.x-p2.x)*(p1.x-p2.x) +
           (p1.y-p2.y)*(p1.y-p2.y) +
           (p1.z-p2.z)*(p1.z-p2.z);
}

StatesGroup::StatesGroup()
{
    rot_end = M3D::Identity();
    pos_end = vel_end = bias_g = bias_a = gravity = Zero3d;

    cov = MD(DIM_STATE,DIM_STATE)::Identity() * INIT_COV;
    cov.block<9,9>(9,9) = MD(9,9)::Identity() * 1e-5;
}


StatesGroup::StatesGroup(const StatesGroup& b) = default;


StatesGroup& StatesGroup::operator=(const StatesGroup& rhs) = default;


StatesGroup
StatesGroup::operator+(const Matrix<double,DIM_STATE,1>& s) const
{
    StatesGroup a(*this);
    a.rot_end = rot_end * Exp(s(0),s(1),s(2));
    a.pos_end += s.block<3,1>(3,0);
    a.vel_end += s.block<3,1>(6,0);
    a.bias_g  += s.block<3,1>(9,0);
    a.bias_a  += s.block<3,1>(12,0);
    a.gravity += s.block<3,1>(15,0);
    return a;
}


StatesGroup&
StatesGroup::operator+=(const Matrix<double,DIM_STATE,1>& s)
{
    rot_end = rot_end * Exp(s(0),s(1),s(2));
    pos_end += s.block<3,1>(3,0);
    vel_end += s.block<3,1>(6,0);
    bias_g  += s.block<3,1>(9,0);
    bias_a  += s.block<3,1>(12,0);
    gravity += s.block<3,1>(15,0);
    return *this;
}


Matrix<double,DIM_STATE,1>
StatesGroup::operator-(const StatesGroup& b) const
{
    Matrix<double,DIM_STATE,1> a;
    M3D rotd(b.rot_end.transpose() * rot_end);

    a.block<3,1>(0,0)  = Log(rotd);
    a.block<3,1>(3,0)  = pos_end - b.pos_end;
    a.block<3,1>(6,0)  = vel_end - b.vel_end;
    a.block<3,1>(9,0)  = bias_g  - b.bias_g;
    a.block<3,1>(12,0) = bias_a  - b.bias_a;
    a.block<3,1>(15,0) = gravity - b.gravity;
    return a;
}


void StatesGroup::resetpose()
{
    rot_end = M3D::Identity();
    pos_end = vel_end = Zero3d;
}