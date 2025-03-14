#ifndef OBJECT_SLAM_OBJECT_REGION_FEATURES_H
#define OBJECT_SLAM_OBJECT_REGION_FEATURES_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentSet.h>
#include <ConcurrentVector.h>

#include <Gaussian/GaussianObject.h>

namespace ObjectSLAM {
	
	/*namespace GOMAP {
		class GaussianObject;
		class GO2D;
	}*/

	class FrameInstance;
	class InstanceMask;

	class ObjectRegionFeatures {
	public:
		ObjectRegionFeatures() :mpRefMap(nullptr), mpRefIns(nullptr) {}
		ObjectRegionFeatures(GOMAP::GaussianObject* _map, FrameInstance* _ins) :mpRefMap(_map), mpRefIns(_ins){}
		virtual ~ObjectRegionFeatures() {}
	public:
		GOMAP::GaussianObject* mpRefMap;
		FrameInstance* mpRefIns;
		GOMAP::GO2D map2D;
		cv::RotatedRect region;               // 객체 영역 or 불확실성
		cv::Rect rect;
		cv::Mat mask;
		std::vector<cv::KeyPoint> keypoints;  // 영역 내 키포인트
		cv::Mat descriptors;                  // 키포인트에 대응하는 디스크립터
		//frame id 필요
	};
}
#endif  