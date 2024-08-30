#ifndef OBJECT_SLAM_OBJECT_MAP_H
#define OBJECT_SLAM_OBJECT_MAP_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

#include <AbstractPose.h>

namespace ObjectSLAM {
	class BoundingBox;
	class ObjectMap {
	public:
		ObjectMap(){
			mpObjectPose = new BaseSLAM::AbstractPose();
			mpWorldPose = new BaseSLAM::AbstractPose();
		}
		virtual~ObjectMap(){
			delete mpObjectPose;
			delete mpWorldPose;
			vecBoundingBoxes.Release();
		}
		//키프레임
		//맵포인트
	public:
		BaseSLAM::AbstractPose *mpObjectPose, *mpWorldPose;
		ConcurrentVector<BoundingBox*> vecBoundingBoxes;
	};

	//같은 레이블 내에서 인스턴스 별로 맵을 관리
	class ObjectInstanceMap {
	public:

	};

	//오브젝트 레이블에서 맵을 관리
	class ObjectLabelMap {
	public:
	};

}
#endif