#ifndef OBJECT_SLAM_GAUSSIAN_VISUALIZER_H
#define OBJECT_SLAM_GAUSSIAN_VISUALIZER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

namespace ObjectSLAM {

	namespace GOMAP {
		class GaussianObject;
	}

	class GaussianVisualizer{
	public:
		static void visualize2D(cv::Mat& image,
			GOMAP::GaussianObject* pGO,
			const cv::Mat& K,
			const cv::Mat& R,
			const cv::Mat& t,
			const cv::Scalar& color = cv::Scalar(0, 0, 255),
			float confidence = 0.95f,
			int thickness = 2);
		static void visualize3D(cv::Mat& image,
			GOMAP::GaussianObject * pGO,
			const cv::Mat& K,
			const cv::Mat& R,
			const cv::Mat& t,
			const cv::Scalar& color = cv::Scalar(255, 255, 0),
			float scale = 1.0f);
	private:
		static std::vector<cv::Point3f> generateEllipsoidPoints(const cv::Mat& covariance,
			const cv::Point3f& center,
			float scale = 1.0f,
			int resolution = 20);
	};
}
#endif