#ifndef OBJECT_SLAM_GAUSSIAN_MAP_MANAGER_H
#define OBJECT_SLAM_GAUSSIAN_MAP_MANAGER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

namespace EdgeSLAM {
	class KeyFrame;
}

namespace ObjectSLAM {

	
	class FrameInstance;

	namespace GOMAP {
		class GaussianObject;
	}

	class GaussianMapManager{
	public:
		static bool InitializeObject(GOMAP::GaussianObject* pG);
		static bool InitializeObject(GOMAP::GaussianObject* pG, FrameInstance* pPrev, FrameInstance* pCurr);
		static GOMAP::GaussianObject* InitializeObject(FrameInstance* pPrev, FrameInstance* pCurr);
		static void UpdateObjectWithIncremental(GOMAP::GaussianObject* pGO, FrameInstance* pCurr);
		static void UpdateObjectWithEKF(GOMAP::GaussianObject* pGO, FrameInstance* pCurr);
	private:
		static const int min_observations = 2;

		static void pointToRay(cv::Mat& ray, const cv::Point2f& pt, const cv::Mat& Rwc, float fx, float fy, float cx, float cy);
		static bool compute3DLineIntersection(
			const cv::Point3f& line1_point, const cv::Point3f& line1_direction,
			const cv::Point3f& line2_point, const cv::Point3f& line2_direction,
			cv::Point3f& intersection, float& distance);
		static bool CheckObjectPosition(const cv::Mat& X, cv::Point2f pt, const cv::Mat& R, const cv::Mat& t
			, float fx, float fy, float cx, float cy, float sigmaSquare = 1.0);
		static bool triangulatePoint(const cv::Mat& xn1, const cv::Mat& xn2, const cv::Mat& Tcw1, const cv::Mat& Tcw2, cv::Mat& x3D);
		static int computeCovariance(cv::Mat& cov, FrameInstance* pIns, 
			const cv::Mat& Tcw, const cv::Mat& Rwo, const cv::Mat& mean, float invfx, float invfy);
		static void computeCovariance(cv::Mat& cov, const cv::Rect& rect, const cv::Mat& pos, const cv::Mat& center, float invfx, float invfy);
		static void computeJacobian(cv::Mat& j, const cv::Mat& R, const cv::Mat& X, float fx, float fy);
	};
}
#endif