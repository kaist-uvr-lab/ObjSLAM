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
	class SemanticConfLabel;
}

namespace ObjectSLAM {
	//class ObjectSLAM;
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

		void InitInstance(const cv::Mat& mapInstance);
		void Init();
		void BaseObjectRegistration(EdgeSLAM::KeyFrame* pNewKF);

		void UpdateInstanceKeyPoints(const std::vector<std::pair<cv::Point2f, cv::Point2f>>& vecPairPoints, const std::vector<std::pair<int, int>>& vecMatches, std::map < std::pair<int, int>, std::pair<int, int>>& mapChangedIns);
		void UpdateInstanceKeyPoints(const std::vector<std::pair<int, int>>& vecMatches, const std::vector<int>& vecIDXs, std::map < std::pair<int, int>, std::pair<int, int>>& mapChangedIns);
		void UpdateInstances(BoxFrame* pTarget, const std::map < std::pair<int, int>, std::pair<int, int>>& mapChanged);
		void UpdateInstances(BoxFrame* pTarget, const std::map<int,int>& mapLinkIDs);
		
		void MatchingFrameWithDenseOF(BoxFrame* pTarget, std::vector<cv::Point2f>& vecPoints1, std::vector<cv::Point2f>& vecPoints2, int scale = 1);
		void MatchingWithFrame(EdgeSLAM::Frame* pTarget, const cv::Mat& fgray, std::vector<int>& vecInsIDs, std::map<int, int>& mapInsNLabel, std::vector<cv::Point2f>& vecCorners);
		void MatchingWithFrame(BoxFrame* pTarget, std::vector<int>& vecIDXs, std::vector<std::pair<int, int>>& vecPairMatches, std::vector<std::pair<cv::Point2f, cv::Point2f>>& vecPairVisualizedMatches);
		void MatchingWithFrame(const cv::Mat& image, const cv::Mat& T, const cv::Mat& K2, std::vector<int>& vecIDXs, std::vector<std::pair<int, cv::Point2f>>& vecPairMatches);

		int GetFrameInstanceId(EdgeSLAM::MapPoint* pMP);
		SegInstance* GetFrameInstance(EdgeSLAM::MapPoint* pMP);

		void InitLabelCount(int N = 200);
		cv::Mat matLabelCount;

		int GetInstance(const cv::Point& pt);
		void SetInstance(const cv::Point& pt, int _sid);

	public:
		BaseSLAM::BaseDevice* mpDevice;
		EdgeSLAM::KeyFrame* mpRefKF;
		BoxFrame* mpPrevBF;
		//yolo
		std::vector<BoundingBox*> mvpBBs;
		//detectron2
		std::map<int, SegInstance*> mmpBBs;
		
		//키포인트에 인스턴스 id를 빠르게 연결
		std::vector<int> mvnInsIDs;
		//std::vector<EdgeSLAM::SemanticConfLabel*> mvpConfLabels;

		cv::Mat img, gray, edge;
		cv::Mat depth;
		cv::Mat labeled;
		cv::Mat mUsed;
		//cv::Mat origin;  

		std::atomic<int> mnMaxID;

		std::atomic<bool> mbInitialized;
		//처음에 초기화로만?
		std::map<int, cv::Mat> sinfos;
	private:
		cv::Mat seg;
		std::mutex mMutexInstance;
	public:
		//매칭 가능한 정보들 추가
		BaseSLAM::KeyPointContainer* mpKC;
		BaseSLAM::StereoDataContainer* mpSC;
	};
}
#endif