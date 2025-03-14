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
	namespace GOMAP {
		class GaussianObject;
	}
	class FrameInstance {
	public:
		FrameInstance(EdgeSLAM::KeyFrame* _pKF, InstanceType _type = InstanceType::SEG) : mpRefKF(_pKF), area(0.0), type(_type), mpGO(nullptr){}
		virtual ~FrameInstance() {}

	public:
		FrameInstance* ConvertedInstasnce(EdgeSLAM::KeyFrame* pKF, cv::Point2f pt);
	public:
		//matching ������ ���ԵǾ�� ��.
		//���� ����ȭ�� ����
		EdgeSLAM::KeyFrame* mpRefKF;
		cv::Mat mDescriptor;
		std::vector<cv::KeyPoint> mvKeys, mvKeyUns;

		void Update(EdgeSLAM::KeyFrame* pKF);
		
		//GlobalInstance* mpGlobal;
		std::set<EdgeSLAM::MapPoint*> setMPs;
		std::set<int> setKPs;

		//����
		InstanceType type; // 1 : seg, 2 : sam, 3 : RAFT, 4 : map
		std::vector<cv::Point> contour;
		cv::Mat mask;
		cv::Rect rect;
		cv::RotatedRect rrect;//elliipse and rotated rect;
		cv::Point2f pt;
		float area;

		//additional
		GOMAP::GaussianObject* mpGO;

	};
}

#endif