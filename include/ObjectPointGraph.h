#ifndef OBJECT_SLAM_OBJECT_POINT_GRAPH_H
#define OBJECT_SLAM_OBJECT_POINT_GRAPH_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

#include <CovisibilityGraph.hpp>

namespace EdgeSLAM {
	class MapPoint;
}
namespace ObjectSLAM {
	class BoundingBox;
	class SegInstance;
	class ObjectPoint;
	class ObjectPointGraph : public BaseSLAM::CovisibilityGraph<BoundingBox, ObjectPoint>{
	public:
		//이 둘을 하나로 합쳐야 할 듯.
		BoundingBox* mpBox;
		SegInstance* mpInstance;

		void SetBadFlag();
		void UpdateConnections(int th = 10);
		void UpdateLocalMap(std::vector<BoundingBox*>& vpLocalKFs, std::vector<ObjectPoint*>& vpLocalMapDatas);
		void UpdateKeyFrames(std::vector<BoundingBox*>& vpLocalKFs);
		void UpdateLocalMapDatas(const std::vector<BoundingBox*>& vpLocalKFs, std::vector<ObjectPoint*>& vpLocalMapDatas);
	private:
		
	};
}

#endif
