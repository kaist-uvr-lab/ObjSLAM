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

	class AssoMatchingData {
	public:
		AssoMatchingData(){}
		virtual ~AssoMatchingData(){}

		std::map<int, int> mapPrevAsso, mapCurrAsso, mapSAM;
		std::map<int, AssociationResultType> mapPrevResult, mapCurrResult; //prev���� r

	};

	class AssoFramePairData {
	public:
		AssoFramePairData():fromid(-1),toid(-1),mpFrom(nullptr),mpTo(nullptr), mpRaftIns(nullptr), mpSamIns(nullptr)
			, mpSamIns2(nullptr), mpPrevMapIns(nullptr), mpLocalMapIns(nullptr), mpFrameMapIns(nullptr)
		, mRaftAssoData(nullptr), mSamAssoData(nullptr) , mFrameMapAssoData(nullptr) , mLocalMapAssoData(nullptr) {}
		AssoFramePairData(BoxFrame* pFrom, BoxFrame* pTo);
		virtual ~AssoFramePairData() {}

	public:
		int fromid, toid;
		BoxFrame *mpFrom, *mpTo;

		void SetFromFrame(BoxFrame* pF);
		void SetToFrame(BoxFrame* pF);

		InstanceMask *mpRaftIns, *mpSamIns, *mpSamIns2, *mpPrevMapIns, *mpLocalMapIns, *mpFrameMapIns; //RAFT�� SAM + tracking�� ���⿡ ����ϱ�
		AssoMatchingData *mRaftAssoData, * mSamAssoData, *mFrameMapAssoData, *mLocalMapAssoData;

		//                                    prev(from)             curr(to)
		//std::set<int> setSegFromFailed, setSamFromFailed, setSegToFailed, setSamToFailed;

		//1) sam request�� raft ����, 2) raft�� to�� segmentation ��Ī, 3) to�� sam�� raft ��Ī, 4)to�� sam�� seg ��Ī
		//std::map<int, int> mapReqRaft, mapRaftSeg, mapRaftSam, mapSamRaft, mapSamSeg, mapMapSeg, mapMapSam, mapLocalMapSeg, mapLocalMapSam, mapFrameMapSeg, mapFrameMapSam;
		//std::map<int, AssociationResultType> mapRaftResult, mapPrevMapResult, mapLocalMapResult, mapFrameMapResult;
		//reqraft, raftseg���� prev���� ��Ī ���� id ȹ��
	};
}

#endif