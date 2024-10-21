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
	class ObjectMatchingInfo;
	enum class ObjectMapState {
		NotEstimated, Success, Failed
	};
	class ObjectSLAM {
	public:
		ObjectSLAM() : mbUsedYolo(false), mbUsedSam2(false), mbUsedDetectron2(false){
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
		
		std::mutex mMutexMatches;
		std::map<std::pair<int, int>, ObjectMatchingInfo*> MapMatches;
		void SetMatchInfo(int target, int ref, ObjectMatchingInfo* pInfo) {
			std::unique_lock<std::mutex> lock(mMutexMatches);
			MapMatches[std::make_pair(target, ref)] = pInfo;
		}
		ObjectMatchingInfo* GetMatchInfo(int target, int ref) {
			auto pair = std::make_pair(target, ref);
			std::unique_lock<std::mutex> lock(mMutexMatches);
			if (MapMatches.count(pair))
				return MapMatches[pair];
			return nullptr;
		}

		void GetAllMatchInfos(std::vector<std::pair<int, int>>& vecPair) {
			std::unique_lock<std::mutex> lock(mMutexMatches);
			for (auto pair : MapMatches) {
				vecPair.push_back(pair.first);
			}
		}

		std::atomic<bool> mbUsedYolo, mbUsedSam2, mbUsedDetectron2;

		/*int SearchMatchInfo(int id, bool bTarget = true) {
			bool bres = false;
			int idx = -1;
			std::unique_lock<std::mutex> lock(mMutexMatches);
			for (auto pair : MapMatches) {
				int target = pair.first.first;
				int reference = pair.first.second;
				if (bTarget) {
					if(target == id)
				}
			}
			return idx;
		}*/
		
	};
}
#endif