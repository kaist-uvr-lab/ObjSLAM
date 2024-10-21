#ifndef OBJECT_SLAM_MATCHING_INFO_H
#define OBJECT_SLAM_MATCHING_INFO_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

namespace ObjectSLAM {
	class ObjectMatchingInfo {
	public:
		ObjectMatchingInfo() : mnTargetId(0), mnReferenceId(0) {}
		ObjectMatchingInfo(int _t, int _r):mnTargetId(_t), mnReferenceId(_r) {}
		virtual ~ObjectMatchingInfo() {
			std::vector<uchar>().swap(vecFounds);
			std::vector<cv::Point2f>().swap(vecTargetCorners);
			std::vector<cv::Point2f>().swap(vecReferenceCorners);
		}
	public:
		int mnTargetId, mnReferenceId;
		std::vector<uchar> vecFounds;
		std::vector<int> vecRefIdxs, vecTargetIdxs;

		std::vector<cv::Point2f> vecTargetCorners, vecReferenceCorners;
	};
}


#endif