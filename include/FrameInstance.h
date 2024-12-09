#ifndef OBJECT_SLAM_FRAME_INSTANCE_H
#define OBJECT_SLAM_FRAME_INSTANCE_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

#include <MapPoint.h>
#include <ObjectSLAMTypes.h>

namespace EdgeSLAM {
	class KeyFrame;
}

namespace ObjectSLAM {
	class FrameInstance {
	public:
		FrameInstance() : area(0.0) {}
		virtual ~FrameInstance() {}

	public:
		//matching 정보가 포함되어야 함.
		//포즈 최적화를 위해

		void Update(EdgeSLAM::KeyFrame* pKF);
		
		//GlobalInstance* mpGlobal;
		std::set<EdgeSLAM::MapPoint*> setMPs;
		std::set<int> setKPs;

		std::vector<cv::Point> contour;
		cv::Mat mask;
		cv::Rect rect;
		cv::RotatedRect rrect;//elliipse and rotated rect;
		cv::Point2f pt;
		float area;
	};
}

#endif