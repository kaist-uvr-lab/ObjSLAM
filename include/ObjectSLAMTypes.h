#ifndef OBJECT_SLAM_TYPES_H
#define OBJECT_SLAM_TYPES_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

#include <BaseSystem.h>
#include <DataContainer.h>

#include <MapDataContainer.h>
#include <Observation.h>

#include <MapPoint.h>

namespace EdgeSLAM {
	class MapPoint;
}

namespace ObjectSLAM {

	enum class InstanceType {
		SEG, SAM, RAFT, MAP
	};
	enum class AssociationResultType {
		SUCCESS, FAIL, RECOVERY
	};

	class ObjectPoint;
	class BoundingBox;
	typedef BaseSLAM::MapDataContainer<ObjectPoint> ObjPointContainer;
	typedef BaseSLAM::Observation<BoundingBox, ObjPointContainer> BoxObjObservation;
	typedef BaseSLAM::MapDataContainer<EdgeSLAM::MapPoint> MapPointContainer;
	typedef BaseSLAM::Observation<BoundingBox, MapPointContainer> BoxMPObservation;

}
#endif