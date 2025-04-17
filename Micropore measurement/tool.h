#pragma once
#include "best_fit.h"
#include <fstream>
#include <string>
#include "libxl.h"
#include <random>
#include <numeric>
#include <sstream>
#include <thread>
#include <mutex>
#include <future>
#include "nanoflann.hpp"
#include <condition_variable>
#include <atomic>  // For std::atomic_bool
random_device rd;
mt19937 gen(rd());

static Vector3d z_axis(0.0, 0.0, 1.0);

double compute_angle_between_vectors(Vector3d& v1, Vector3d& v2)//����Ƕ���
{
	double dot_product = v1.dot(v2);
	double norm_v1 = v1.norm();
	double norm_v2 = v2.norm();
	double cos_theta = dot_product / (norm_v1 * norm_v2);

	cos_theta = max(min(cos_theta, 1.0), -1.0); // ȷ�� cos_theta �� [-1, 1] ��Χ��

	double theta_radians = acos(cos_theta); // �����Һ������õ��нǵĻ���ֵ

	// ת��Ϊ�Ƕ�ֵ
	double theta_degrees = theta_radians * 180 / M_PI;

	if (theta_degrees > 90 && theta_degrees < 180)
	{
		theta_degrees = 180 - theta_degrees;
	}

	return theta_degrees;

}

Point3d sample_elliptical_cylinder(const Matrix3d& covariance_matrix, const Point3d& center)
{
	LLT<Matrix3d> llt(covariance_matrix); //Cholesky�ֽ�
	Matrix3d L = llt.matrixL();//�����Ǿ���

	normal_distribution<double> dis(0.0, 1.0);

	Point3d point(center.size());
	for (int j = 0; j < 3; j++)
	{
		point(j) = dis(gen);
	}
	return center + L * point;
}

Points3d readPointsFromFile(const string& filename)
{
	ifstream file(filename);
	if (!file.is_open())
	{
		throw std::runtime_error("Error: unable to open file");
	}

	string lines;
	Points3d points;
	while (getline(file, lines))
	{
		stringstream ss(lines);
		Point3d p;
		ss >> p(0) >> p(1) >> p(2);
		points.push_back(p);
	}
	file.close();
	return points;
}

Vector3d get_plane_normal(const Points3d& points)//����֤��ȷ
{
	//��������
	Point3d points_centroid = Point3d::Zero();
	for (auto& p : points)
	{
		points_centroid += p;
	}
	points_centroid /= points.size();
	//���Ļ��㼯
	Points3d centered_points;
	for (auto& p : points)
	{
		centered_points.emplace_back(p - points_centroid);
	}
	//��centered_points�任ΪN*3�ľ���
	MatrixXd centered_points_matrix(centered_points.size(), 3);

	for (int i = 0; i < centered_points.size(); i++)
	{
		centered_points_matrix.row(i) = centered_points[i].transpose();
	}
	JacobiSVD<MatrixXd> svd(centered_points_matrix, ComputeFullV);
	Vector3d normal = svd.matrixV().col(2);

	return normal;
}

// �����ֵ
double calculateMean(const std::vector<double>& data) {
	if (data.empty()) return 0.0;
	double sum = std::accumulate(data.begin(), data.end(), 0.0);
	return sum / data.size();
}

// �����׼��
double calculateStandardDeviation(const std::vector<double>& data, double mean) {
	if (data.size() <= 1) return 0.0;
	double varianceSum = 0.0;
	for (const auto& value : data) {
		varianceSum += (value - mean) * (value - mean);
	}
	return std::sqrt(varianceSum / (data.size() - 1)); // ʹ��������׼�ʽ
} 

class PointCloud
{
public:
	// Ϊ nanoflann ���� Vector3d �ĵ��ƽṹ
	Points3d pts;
	inline size_t kdtree_get_point_count() const { return pts.size(); }
	inline double kdtree_get_pt(const size_t idx, const size_t dim) const { return pts[idx](dim); }

	// �������ݼ��ı߽��
	template <class BBOX>
	bool kdtree_get_bbox(BBOX&) const
	{
		return false;
	}
};
//��Ӷ�ƽ�����ݽ������������������ĺ���
Points3d laHySample(const Points3d& points, int n)
{
	vector<double> minVal = { points[0](0), points[0](1), points[0](2) };
	vector<double> maxVal = minVal;
	for (const auto& p : points)
	{
		minVal[0] = min(minVal[0], p[0]);
		minVal[1] = min(minVal[1], p[1]);
		minVal[2] = min(minVal[2], p[2]);
		maxVal[0] = max(maxVal[0], p[0]);
		maxVal[1] = max(maxVal[1], p[1]);
		maxVal[2] = max(maxVal[2], p[2]);
	}

	Points3d samples(n);
	for (int dim = 0; dim < 3; dim++)
	{
		vector<double> intervals(n);
		double step = (maxVal[dim] - minVal[dim]) / n;
		for (int i = 0; i < n; i++)
		{
			uniform_real_distribution<> dis(minVal[dim] + i * step, minVal[dim] + (i + 1) * step);
			intervals[i] = dis(gen);
		}
		shuffle(intervals.begin(), intervals.end(), gen);
		for (int i = 0; i < n; i++)
		{
			samples[i](dim) = intervals[i];
		}
	}
	PointCloud pc;
	pc.pts = points;
	typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud>, PointCloud, 3> KDTree;
	KDTree tree(3, pc, nanoflann::KDTreeSingleIndexAdaptorParams(10));
	tree.buildIndex();

	Points3d selected_points;
	selected_points.reserve(n);
	for (const auto& s : samples)
	{
		array<double, 3> query = { s[0], s[1], s[2] };
		size_t ret_idx;
		double out_dist_sqr;

		nanoflann::KNNResultSet<double> resultSet(1);
		resultSet.init(&ret_idx, &out_dist_sqr);
		tree.findNeighbors(resultSet, query.data(), nanoflann::SearchParameters());
		selected_points.emplace_back(pc.pts[ret_idx]);
	}
	return selected_points;
}

Matrix3d compute_rotation_matrix(const Vector3d& direction, double angle)
{
	/*
		direction:����������Բ��Ϊ��������ƽ��Ϊ��������
		angle:����ת��ļнǣ���λ�����ȣ�
		һ�㽫Բ������������ת����������Z��ƽ���ٴ���ƽ������Ҳ����һ��ѡ�����λ�ñ��ֲ���
	*/
	Vector3d axis = direction.normalized();
	Vector3d axis_z(0.0, 0.0, 1.0);
	Vector3d n = direction.cross(axis_z).normalized();
	Matrix3d n_cross  { {0,-n[2],n[1]}, {n[2],0,-n[0]}, {-n[1],n[0],0} };
	Matrix3d I = Matrix3d::Identity();
	Matrix3d R = I + sin(angle) * n_cross + (1 - cos(angle)) * n_cross * n_cross;
	return R;
}

pair<Points3d, Points3d> oridataprocess(const string &yzfilepath,const string &pmfilepath,int pmdata_select_nums)//ԭʼ���ݴ���ɳ������������ʽ
{
	Points3d yzdata, pmdata;
	yzdata = readPointsFromFile(yzfilepath);
	pmdata = readPointsFromFile(pmfilepath);

	tuple<Vector3d, Point3d, double, double> t = _best_fit(yzdata);
	Vector3d yz_direction = get<0>(t);
	double angle_degree = compute_angle_between_vectors(yz_direction, z_axis);
	Matrix3d R_yz = compute_rotation_matrix(yz_direction, angle_degree * M_PI / 180.0);//���㽫Բ��������תΪZ�����ת����

	//Բ��������ת��Z��ƽ�У�ƽ�����ݸ���һ����ת�����λ�ñ��ֲ���
	Points3d yzdata_parallel_z;
	Points3d pmdata_follow_yz;
	thread t3([&yzdata , &R_yz, &yzdata_parallel_z]()
		{
			for (const auto& p : yzdata)
			{
				Point3d p_parallel_z = R_yz * p;
				yzdata_parallel_z.emplace_back(p_parallel_z);
			}
		});
	thread t4([&pmdata ,&pmdata_follow_yz,&R_yz,&pmdata_select_nums]()
		{
			Points3d pmdata_select = laHySample(pmdata, pmdata_select_nums);
			for (const auto& p : pmdata_select)
			{
				Point3d p_follow_yz = R_yz * p;
				pmdata_follow_yz.emplace_back(p_follow_yz);
			}
		});
	t3.join();
	t4.join();

	return make_pair(yzdata_parallel_z, pmdata_follow_yz);
}

pair<double, double > get_angle_and_diameter(Points3d& yzdata_parallel_z, Points3d& pmdata_follow_yz)
{
	Vector3d yz_direction(0.0,0.0,0.0);
	Vector3d pm_direction(0.0,0.0,0.0);
	double diameter = 0.0;
	double angle_degree = 0.0;

	// ���� promise �� future
	std::promise<std::tuple<Vector3d, double>> p1;
	std::future<std::tuple<Vector3d, double>> f1 = p1.get_future();
	std::promise<Vector3d> p2;
	std::future<Vector3d> f2 = p2.get_future();

	// �߳� t1
	std::thread t1([&p1, &yzdata_parallel_z]() {
		try {
			auto t = _best_fit(yzdata_parallel_z);
			p1.set_value(std::make_tuple(std::get<0>(t), std::get<3>(t) * 2));
		}
		catch (...) {
			p1.set_exception(std::current_exception());
		}
		});

	// �߳� t2
	std::thread t2([&p2, &pmdata_follow_yz]() {
		try {
			p2.set_value(get_plane_normal(pmdata_follow_yz));
		}
		catch (...) {
			p2.set_exception(std::current_exception());
		}
		});

	t1.join();
	t2.join();

	// ��ȡ����������쳣
	auto [yz_dir, dia] = f1.get(); // �׳��쳣����д���
	yz_direction = yz_dir;
	diameter = dia;
	pm_direction = f2.get(); // �׳��쳣����д���

	angle_degree = compute_angle_between_vectors(yz_direction, pm_direction);
	return make_pair(angle_degree, diameter);
}

pair<Points3d, Points3d> sample_yz_pm(Points3d& yzdata_parallel_z, Points3d& pmdata_follow_yz,const Matrix3d& R_pm,
										const Matrix3d& covariance_matrix_yz ,const Matrix3d& covariance_matrix_pm)
{
	Points3d yzdata_sampled, pmdata_sampled;

	//[]������ Lambda ����ʱ��δ��ⲿ�����롱���ݣ��� ()������ Lambda ����ʱ��Ҫ����ʲô����
	thread t1([&yzdata_parallel_z, &covariance_matrix_yz, &yzdata_sampled]()
		{
			for (const auto& p : yzdata_parallel_z)
			{
				Point3d p_sampled = sample_elliptical_cylinder(covariance_matrix_yz, p);
				yzdata_sampled.emplace_back(p_sampled);
			}
		});
	thread t2([ &pmdata_follow_yz, &R_pm, &covariance_matrix_pm, &pmdata_sampled]()
		{
			for (const auto& p : pmdata_follow_yz)
			{
				Point3d p_sampled = R_pm.transpose() * sample_elliptical_cylinder(covariance_matrix_pm, p);
				pmdata_sampled.emplace_back(p_sampled);
			}
		});
	t1.join();
	t2.join();
	return make_pair(yzdata_sampled, pmdata_sampled);
}

void sampler_thread_func(int sample_nums)//�����߳̿��ƺ���
{

}