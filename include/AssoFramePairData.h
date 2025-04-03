#ifndef OBJECT_SLAM_ASSO_FRAME_PAIR_DATA_H
#define OBJECT_SLAM_ASSO_FRAME_PAIR_DATA_H
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
	class BoxFrame;
	class InstanceMask;

	namespace GOMAP {
		class GaussianObject;
	}
	class AssoFramePairData {
	public:
		AssoFramePairData():fromid(-1),toid(-1),mpFrom(nullptr),mpTo(nullptr), mpRaftIns(nullptr), mpSamIns(nullptr), mpPrevMapIns(nullptr), mpLocalMapIns(nullptr){}
		AssoFramePairData(BoxFrame* pFrom, BoxFrame* pTo);
		virtual ~AssoFramePairData() {}

	public:
		int fromid, toid;
		BoxFrame *mpFrom, *mpTo;

		InstanceMask *mpRaftIns, *mpSamIns, *mpPrevMapIns, *mpLocalMapIns; //RAFT와 SAM을 여기에 기록하기
		//                                    prev(from)             curr(to)
		//1) sam request시 raft 정보, 2) raft와 to의 segmentation 매칭, 3) to의 sam과 raft 매칭, 4)to의 sam과 seg 매칭
		std::map<int, int> mapReqRaft, mapRaftSeg, mapRaftSam, mapSamRaft, mapSamSeg, mapMapSeg, mapMapSam, mapLocalMapSeg, mapLocalMapSam;
		std::set<int> setSegFromFailed, setSamFromFailed, setSegToFailed, setSamToFailed;
		std::map<int, AssociationResultType> mapRaftResult, mapPrevMapResult, mapLocalMapResult;
		//reqraft, raftseg에서 prev에서 매칭 실패 id 획득
	};
}

#endif