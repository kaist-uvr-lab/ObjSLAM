#ifndef OBJECT_SLAM_OBJECT_INITIALIZER_H
#define OBJECT_SLAM_OBJECT_INITIALIZER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

namespace ObjectSLAM {
	class ObjectMap;
	class BoundingBox;
	class ObjectInitializer{
	public:
		ObjectInitializer() {}
		virtual~ObjectInitializer() {
		}
		//키프레임
		//맵포인트
	public:
		bool Initialization();
		//bool StereoInitialization(BoundingBox* pBox, ObjectMap* pObjMap);
		ObjectMap* StereoInitialization(BoundingBox* pBox);
		bool MonocularInitialization();
	};
}
#endif