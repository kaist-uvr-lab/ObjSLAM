#ifndef OBJECT_SLAM_OBJECT_REGION_FEATURES_H
#define OBJECT_SLAM_OBJECT_REGION_FEATURES_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentSet.h>
#include <ConcurrentVector.h>

#include <Gaussian/GaussianObject.h>

namespace EdgeSLAM{
	class MapPoint;
}

namespace ObjectSLAM {
	
	/*namespace GOMAP {
		class GaussianObject;
		class GO2D;
	}*/

	class FrameInstance;
	class InstanceMask;
	class BoxFrame;

	class ObjectRegionFeatures {
	public:
		ObjectRegionFeatures() :mpRefMap(nullptr), mpRefIns(nullptr), mpRefFrame(nullptr ) {}
		ObjectRegionFeatures(GOMAP::GaussianObject* _map, FrameInstance* _ins, BoxFrame* _box = nullptr) :mpRefMap(_map), mpRefIns(_ins), mpRefFrame(_box){}
		virtual ~ObjectRegionFeatures() {}
	public:
		GOMAP::GaussianObject* mpRefMap;
		FrameInstance* mpRefIns;
		BoxFrame* mpRefFrame;
		GOMAP::GO2D map2D;
		cv::RotatedRect region;               // ��ü ���� or ��Ȯ�Ǽ�
		cv::Rect rect;
		cv::Mat mask;
		std::vector<EdgeSLAM::MapPoint*> mappoints; //�� �׽�Ʈ��
		std::vector<cv::KeyPoint> keypoints;  // ���� �� Ű����Ʈ
		cv::Mat descriptors;                  // Ű����Ʈ�� �����ϴ� ��ũ����
		//frame id �ʿ�
	};
}
#endif  