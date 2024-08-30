#ifndef OBJECT_SLAM_OPTIMIZER_H
#define OBJECT_SLAM_OPTIMIZER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

namespace ObjectSLAM {
	class ObjectOptimizer {
		static int ObjectPoseOptimization(const std::vector<cv::Point2f>& imgPts, const std::vector<cv::Point3f>& objPts, std::vector<bool>& outliers, cv::Mat& P, const float fx, const float fy, const float cx, const float cy);
	};
}
#endif