  #ifndef OBJECT_SLAM_SLAM_SYSTEM_H
#define OBJECT_SLAM_SLAM_SYSTEM_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

namespace BaseSLAM {
	class BaseDevice;
}
namespace EdgeSLAM {
	class KeyFrame;
	class MapPoint;
}

namespace ObjectSLAM {
	class ObjectMap;
	class ObjectDevice;
	class BoxFrame;
	enum class ObjectMapState {
		NotEstimated, Success, Failed
	};
	class ObjectSLAM {
	public:
		ObjectSLAM() {

		}
		virtual~ObjectSLAM() {
			ObjectMaps.Release();
		}
		void UpdateMapPoint(BoxFrame* pF);
		std::vector<BoxFrame*> GetConnectedBoxFrames(EdgeSLAM::KeyFrame* pKF, int nn = 20);
	public:
		ConcurrentMap<BaseSLAM::BaseDevice*, ObjectDevice*> MapObjectDevices;
		ConcurrentMap<int, BoxFrame*> MapKeyFrameNBoxFrame;
		ConcurrentMap<int, ObjectMap*> ObjectMaps;
	};
}
#endif