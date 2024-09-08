#ifndef OBJECT_SLAM_SEGMENT_INSTANCE_H
#define OBJECT_SLAM_SEGMENT_INSTANCE_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

#include <MapPoint.h>
#include <BaseSystem.h>
#include <MapDataContainer.h>
#include <DataContainer.h>
#include <KeyPointContainer.h>
#include <StereoDataContainer.h>
#include <BaseDevice.h>
#include <AbstractPose.h>
#include <ObjectSLAMTypes.h>

namespace EdgeSLAM {
	class SemanticConfLabel;
}

namespace ObjectSLAM {

	class BoxFrame;
	class ObjectMap;
	class ObjectPointGraph;

	class SegInstance : public BaseSLAM::AbstractFrame, public BaseSLAM::KeyPointContainer, public BaseSLAM::StereoDataContainer, public ObjPointContainer {
	public:
		SegInstance() {}
		SegInstance(BoxFrame* _ref, int _fx, int _fy, int _cx, int _cy, int _label, float _conf, bool _thing, BaseSLAM::BaseDevice* Device, bool _bdetected);
		virtual ~SegInstance() {}
		bool isBad() {
			return mbBad;
		}
		void SetBadFlag() {}
		void StereoDataInit(int N) {
			BaseSLAM::StereoDataContainer::Init(N);
		}
		/*void RemoveDat(int idx) {
			mvKeyDatas.erase(mvKeyDatas.begin()+idx);
			mvKeyDataUns.erase(mvKeyDataUns.begin() + idx);
			mvDepth.erase(mvDepth.begin() + idx);
			mvuRight.erase(mvuRight.begin() + idx);
			mvpMapDatas.Erase(idx);
		}*/
		void AddData(cv::KeyPoint kp, cv::KeyPoint kpun, const cv::Mat& d) {
			BaseSLAM::KeyPointContainer::AddData(kp, d);
			ObjPointContainer::AddMapData(nullptr, false);
			BaseSLAM::StereoDataContainer::AddStereoData(-1.0, -1.0);
			BaseSLAM::KeyPointContainer::AssignFeatureToGrid(N++, kp);
			mvKeyDataUns.push_back(kpun);
			mvbInlierKPs.push_back(true);
		}
		void AddData(cv::KeyPoint kp, const cv::Mat& d) {
			BaseSLAM::KeyPointContainer::AddData(kp, d);
			ObjPointContainer::AddMapData(nullptr, false);
			BaseSLAM::StereoDataContainer::AddStereoData(-1.0, -1.0);
			BaseSLAM::KeyPointContainer::AssignFeatureToGrid(N++, kp);
			if (mpCamera->bDistorted) {
				cv::Mat mat = cv::Mat(kp.pt);
				cv::undistortPoints(mat, mat, mpCamera->K, mpCamera->D, cv::Mat(), mpCamera->K);
				kp.pt = cv::Point2f(mat.at<cv::Vec2f>(0));
			}
			mvKeyDataUns.push_back(kp);
			mvbInlierKPs.push_back(true);
		}

		void AddMapPoint(EdgeSLAM::MapPoint* pMP, const cv::KeyPoint& kp) {
			/*if(!pMP || pMP->isBad()){
				return;
			}*/
			mvpMapPoints.push_back(pMP);
			mvKeyPoints.push_back(kp);
		}

		void Init() {
			BaseSLAM::KeyPointContainer::Init(mpCamera->bDistorted, mpCamera->K, mpCamera->D);
			ObjPointContainer::Reset(N);
			BaseSLAM::StereoDataContainer::Init(N);
		}

		bool isTable();
		bool isFloor();
		bool isTable(int _label, bool _thing);
		bool isFloor(int _label, bool _thing);
		bool isCeiling();
		bool isWall();
		bool isObject();
		bool isStaticObject();

		cv::Mat GetCenter();
		cv::Mat GetPose();
		void SetPose(const cv::Mat& _T);

	public:
		//void Update() override {};
		bool UnprojectStereo(int i, cv::Mat& Xw, cv::Mat& Xo, const cv::Mat& Rwc, const cv::Mat& twc);
	public:
		std::vector<int> mvIDXs;
		std::map<int, int> mapIDXs;
		
		std::vector<cv::Mat> mvWorld, mvObject;
		float fx, fy, cx, cy, invfx, invfy;
		cv::Mat mUsed;
		cv::Mat origin;
		int n1, n2;

		ConcurrentVector<bool> mvbInlierKPs;

		//SLAM MAP°ú ¿¬°á
		std::vector<EdgeSLAM::MapPoint*> mvpMapPoints;
		std::vector<cv::KeyPoint> mvKeyPoints;
		cv::Mat mMatMapDesc;

	public:
		BaseSLAM::AbstractCamera* mpCamera;
		BaseSLAM::AbstractPose* mpWorldPose;
		static std::atomic<int> nSegInstanceId;
		
		std::string mStrLabel;
		bool mbDetected;

		//int mnLabel;
		//float mfConfidence;

		bool mbIsthing;
		cv::Rect mRect;
		BoxFrame* mpRef;

		ObjectMap* mpMap;
		ObjectPointGraph* mpGraph;

		std::atomic<int> mnConnected;
		ConcurrentSet<SegInstance*> mConnectedInstances;
		EdgeSLAM::SemanticConfLabel* mpConfLabel;
		void UpdateInstance(SegInstance* pConnected);
	};
}


#endif