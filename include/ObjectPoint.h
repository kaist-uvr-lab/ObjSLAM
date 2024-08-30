#ifndef OBJECT_SLAM_OBJECT_POINT_H
#define OBJECT_SLAM_OBJECT_POINT_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

#include <AbstractData.h>

#include <ObjectSLAMTypes.h>

namespace ObjectSLAM {
	
	class BoundingBox;
	class SegInstance;

	class ObjectPoint : public BaseSLAM::AbstractData, public BoxObjObservation {
	public:
		ObjectPoint();
		ObjectPoint(BoundingBox* _ref);
		ObjectPoint(SegInstance* _aref);
		virtual ~ObjectPoint(){}
	public:
		void Update() override {

		}
		//업데이트 구현 필요
		/*void UpdateNormalAndDepth();
		void Update() override {
			ComputeDistinctiveDescriptors<ObjectPoint, BaseSLAM::KeyPointContainer, BoxObjObservation>();
			UpdateNormalAndDepth();
		};
		cv::Point2f Projection(const cv::Mat& P);*/

		//computeNormalAndDepth

	public:
		static std::atomic<int> nObjPointId;
		float mfMinDistance;
		float mfMaxDistance;
		cv::Mat mNormalVector;

		BoxObjObservation* mpObservation;
		//getdata,setdata
		//getdesc
		//update
	private:

	};
}

#endif