#ifndef OBJECT_SLAM_BOX_FRAME_H
#define OBJECT_SLAM_BOX_FRAME_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

#include <BaseSystem.h>
#include <KeyPointContainer.h>
#include <StereoDataContainer.h>
#include <BaseDevice.h>

namespace EdgeSLAM {
	class KeyFrame;
	class Frame;
	class MapPoint;
	class SemanticConfidence;
}

namespace ObjectSLAM {
	class BoundingBox;
	class SegInstance;
	class NewBoxFrame {
	public:
		NewBoxFrame(int w, int h):N(0), mUsed(cv::Mat::zeros(h,w, CV_8UC1)), mDesc(cv::Mat::zeros(0,32,CV_8UC1)){

		}
		virtual ~NewBoxFrame() {

		}
		ConcurrentVector<BoundingBox*> mvpBBs;
		std::vector<cv::KeyPoint> mvKeys;
		cv::Mat mDesc;
		cv::Mat mUsed;
		int N;
	private:

	};
	class BoxFrame : public BaseSLAM::AbstractFrame, public BaseSLAM::KeyPointContainer, public BaseSLAM::StereoDataContainer {
	public:
		BoxFrame(int _id);
		BoxFrame(int _id, const int w, const int h, BaseSLAM::BaseDevice* Device, BaseSLAM::AbstractPose* _Pose);
		virtual ~BoxFrame();
		bool isBad() {
			return mbBad;
		}
		void SetBadFlag() {}
		void AddData(cv::KeyPoint kp, const cv::Mat& desc) {
			/*mvKeyDatas.push_back(kp);
			mvbMatched.push_back(false);
			mDescriptors.push_back(desc.clone());
			mUsed.at<uchar>(kp.pt)++;
			AssignFeatureToGrid(N, kp);
			N++;*/
			BaseSLAM::KeyPointContainer::AddData(kp, desc);
			//mUsed.at<uchar>(kp.pt)++;
		}
		
		void Copy(EdgeSLAM::Frame* pF);

		void ConvertInstanceToFrame(std::vector<std::pair<int, int>>& vPairFrameAndBox, std::vector<cv::Point2f>& vecCorners);
		void ConvertBoxToFrame(int w, int h);

		void Init();
		void BaseObjectRegistration(EdgeSLAM::KeyFrame* pNewKF);

		void UpdateInstanceKeyPoints(const std::vector<std::pair<int, int>>& vecMatches, const std::vector<int>& vecIDXs, const std::vector<std::pair<int, int>>& vPairFrameAndBox, std::map < std::pair<int, int>, std::pair<int, int>>& mapChangedIns);
		void UpdateInstances(BoxFrame* pTarget, const std::map < std::pair<int, int>, std::pair<int, int>>& mapChanged);
		void UpdateInstances(BoxFrame* pTarget, const std::map<int,int>& mapLinkIDs);
		void MatchingWithFrame(BoxFrame* pTarget, std::vector<int>& vecIDXs, std::vector<std::pair<int, int>>& vecPairMatches, std::vector<std::pair<int, int>>& vecPairPointIdxInBox);
		void MatchingWithFrame(const cv::Mat& image, const cv::Mat& T, const cv::Mat& K2, std::vector<int>& vecIDXs, std::vector<std::pair<int, cv::Point2f>>& vecPairMatches);

		int GetFrameInstanceId(EdgeSLAM::MapPoint* pMP);
		SegInstance* GetFrameInstance(EdgeSLAM::MapPoint* pMP);

		void InitLabelCount(int N = 200);
		cv::Mat matLabelCount;

	public:
		BaseSLAM::BaseDevice* mpDevice;
		EdgeSLAM::KeyFrame* mpRefKF;
		//yolo
		std::vector<BoundingBox*> mvpBBs;
		//detectron2
		std::map<int, SegInstance*> mmpBBs;
		
		//label vector
		std::vector<int> mvLabels;

		cv::Mat img, gray;
		cv::Mat depth;
		cv::Mat labeled;
		//cv::Mat mUsed;
		//cv::Mat origin;  

		cv::Mat seg;
		std::map<int, cv::Mat> sinfos;
		std::atomic<int> mnMaxID;
	public:
		//매칭 가능한 정보들 추가
		BaseSLAM::KeyPointContainer* mpKC;
		BaseSLAM::StereoDataContainer* mpSC;
	};
}
#endif