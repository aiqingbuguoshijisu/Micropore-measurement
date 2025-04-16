#pragma once
#include <iomanip>
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <nlopt.hpp>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#define M_PI 3.14159265358979323846
using namespace std;
using namespace Eigen;
using namespace nlopt;
using Point3d = Vector3d;
using Points3d = vector<Point3d>;

struct SphericalCoordinates
{
    double theta;
    double phi;
    SphericalCoordinates(double t, double p) : theta(t), phi(p) {}
};

SphericalCoordinates cartesian_to_spherical(float x, float y, float z)
{
    double theta = acos(z / sqrt(x * x + y * y + z * z));
    double phi;
    if (std::abs(x) < 1e-9 && std::abs(y) < 1e-9)
    {
        phi = 0.0;
    }
    else
    {
        phi = copysign(1.0, y) * acos(x / std::sqrt(x * x + y * y));
    }
    return SphericalCoordinates(theta, phi);
}

SphericalCoordinates _compute_initial_direction(const Points3d& pts)
{
    //MatrixXd data(pts.size(), 3);
    //for (int i = 0; i < pts.size(); i++)
    //{
    //    data.row(i) << pts[i][0], pts[i][1], pts[i][2];
    //}
    //MatrixXd centered = data.rowwise() - data.colwise().mean();
    //Matrix3d cov = (centered.transpose() * centered) / double(pts.size() - 1);
    //SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    //Vector3d initial_direction(
    //    eig.eigenvectors().col(2)(0),  // 使用最大特征值对应的特征向量
    //    eig.eigenvectors().col(2)(1),
    //    eig.eigenvectors().col(2)(2)
    //);
    //return cartesian_to_spherical(initial_direction[0], initial_direction[1], initial_direction[2]);
    MatrixXd m(pts.size(), 3);
    for (int i = 0; i < pts.size(); i++)
    {
        m.row(i) = pts[i];
    }
    JacobiSVD<MatrixXd> svd(m, ComputeFullU | ComputeFullV);
    Vector3d initial_direction = svd.matrixV().col(2);
    return cartesian_to_spherical(initial_direction[0], initial_direction[1], initial_direction[2]);
}

Vector3d spherical_to_cartesian(const SphericalCoordinates& spherical_coordinates)
{
    return Vector3d(cos(spherical_coordinates.phi) * sin(spherical_coordinates.theta),
                     sin(spherical_coordinates.phi) * sin(spherical_coordinates.theta),
                    cos(spherical_coordinates.theta));
}

Matrix3d _compute_projection_matrix(const Vector3d& dir)//Vector3d是列向量
{
    return Matrix3d::Identity() - dir*dir.transpose();
}

Matrix3d _compute_skew_matrix(const Vector3d& dir)
{
    Matrix3d skew;
    skew << 0.0, -dir[2], dir[1],
            dir[2], 0.0, -dir[0],
            -dir[1], dir[0], 0.0;
    return skew;
}

Matrix3d _compute_a_matrix(const Points3d& pts)
{
    Matrix3d A;
    for (const auto& pt : pts)
    {
        A += pt * pt.transpose();
    }
    return A /= pts.size();
}

Matrix3d _compute_a_hat_matrix(Matrix3d& A, Matrix3d& skew)
{
    return skew * A * skew.transpose();
}

double _compute_g(const std::vector<double>& x, std::vector<double>& grad, void* data)
{
    SphericalCoordinates sc = { x[0], x[1] };
    Vector3d dir = spherical_to_cartesian(sc);
    Points3d* pts = static_cast<Points3d*>(data);

    Matrix3d projection_matrix = _compute_projection_matrix(dir);
    Matrix3d skew_matrix = _compute_skew_matrix(dir);
    Points3d input_samples;
    input_samples.reserve(pts->size());
    for (const auto& pt : *pts)
    {
        input_samples.push_back(projection_matrix * pt);
    }
    Matrix3d a_matrix = _compute_a_matrix(input_samples);
    Matrix3d a_hat_matrix = _compute_a_hat_matrix(a_matrix, skew_matrix);

    double u = 0.0;
    for (const auto& sample : input_samples)
    {
        u += sample.dot(sample);
    }
    u /= input_samples.size();

    Vector3d sum_v = Vector3d::Zero();
    for (const auto& sample : input_samples)
    {
        sum_v += sample.dot(sample) * sample;
    }
    sum_v /= input_samples.size();

    Vector3d v = a_hat_matrix * sum_v / (a_hat_matrix * a_matrix).trace();

    double error = 0.0;
    int index = 0;
    for (const auto& sample: input_samples)
    {
        double term = sample.dot(sample) - u - 2 * (pts->at(index)).transpose().dot(v);
        error += term * term;
        index++;
    }
    error /= input_samples.size();
    return error;
}

Point3d _points_centroid(const Points3d& pts)
{
    Point3d centroid(0.0, 0.0, 0.0);
    for (const auto& pt : pts)
    {
        centroid += pt;
    }
    centroid /= pts.size();
    return centroid;
}

Points3d _points_centered(const Points3d& pts)
{
    Point3d centroid = _points_centroid(pts);
    Points3d centered_pts;//中心化后的数据
    for (const auto& pt : pts)
    {
        centered_pts.push_back(pt - centroid);
    }
    return centered_pts;
}

Point3d _compute_center(const Vector3d& dir, const Points3d& pts )//pts是中心化后的数据
{
    Matrix3d projection_matrix = _compute_projection_matrix(dir);
    Matrix3d skew_matrix = _compute_skew_matrix(dir);
    Points3d input_samples;
    input_samples.reserve(pts.size());
    for (const auto& pt : pts)
    {
        input_samples.push_back(projection_matrix * pt);
    }

    Matrix3d a_matrix = _compute_a_matrix(input_samples);
    Matrix3d a_hat_matrix = _compute_a_hat_matrix(a_matrix, skew_matrix);
    Vector3d sum_v = Vector3d::Zero();
    for (const auto& sample : input_samples)
    {
        sum_v += sample.dot(sample) * sample;
    }
    sum_v /= input_samples.size();
    Vector3d v = a_hat_matrix * sum_v / (a_hat_matrix * a_matrix).trace();
    return v;//要再加上质心才是center点
}

double _compute_radius(const Vector3d& dir, const Points3d& pts)//ptrs是中心化后的数据
{
    Matrix3d projection_matrix = _compute_projection_matrix(dir);
    Point3d center = _compute_center(dir, pts);

    double sum_radius = 0.0;
    for (const auto& pt : pts)
    {
        Point3d tmp = projection_matrix * (center - pt);
        sum_radius += tmp.dot(tmp);
    }
    double radius = sqrt(sum_radius / pts.size());
    return radius;
}

tuple <Vector3d, Point3d, double,double> _best_fit(const Points3d& pts)//轴向量，中心点，半径
{
    Point3d centroid = _points_centroid(pts);
    Points3d centered_pts = _points_centered(pts);
    SphericalCoordinates x0_ = _compute_initial_direction(centered_pts);
    vector<double> x0 = { x0_.theta, x0_.phi };
    opt opt(LN_PRAXIS,2);
    opt.set_min_objective(_compute_g, const_cast<Points3d*>(&centered_pts));

    vector<double> lb = { -2 * M_PI, -2*M_PI }; // 下界
    vector<double> ub = { 2*M_PI,  2*M_PI }; // 上界
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);

    opt.set_xtol_rel(1e-5);
    opt.set_maxeval(5000);
    double minf;
    try
    {
        result res = opt.optimize(x0, minf);
        SphericalCoordinates best_fit_x0(x0[0], x0[1]);
        Vector3d best_fit_dir = spherical_to_cartesian(best_fit_x0);
        Point3d best_fit_center = _compute_center(best_fit_dir, centered_pts)+centroid;
        double best_fit_radius = _compute_radius(best_fit_dir, centered_pts);
        double best_fit_error = minf;
        return make_tuple(best_fit_dir, best_fit_center, best_fit_radius, best_fit_error);
    }
    catch (exception& e)
    {
        cout << "Optimization failed: " << e.what() << endl;
        return make_tuple(Vector3d::Zero(), Point3d::Zero(), 0.0, 0.0);
    }
}