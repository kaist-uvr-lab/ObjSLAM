#ifndef OBJECT_SLAM_BOUNDING_BOX_H
#define OBJECT_SLAM_BOUNDING_BOX_H
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

namespace ObjectSLAM {
	class BoxFrame;
	class ObjectMap;
	class ObjectPointGraph;

	class BoundingBox : public BaseSLAM::AbstractFrame, public BaseSLAM::KeyPointContainer, public BaseSLAM::StereoDataContainer, public ObjPointContainer{
	public:
		BoundingBox(){}
		BoundingBox(BoxFrame* _ref, int _fx, int _fy, int _cx, int _cy, int _label, float _conf, cv::Point2f left, cv::Point2f right, BaseSLAM::BaseDevice* Device);

		virtual ~BoundingBox(){}
		bool isBad() {
			return mbBad;
		}
		void SetBadFlag() {}
		void StereoDataInit(int N) {
			BaseSLAM::StereoDataContainer::Init(N);
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
		}
		void Init() {
			BaseSLAM::KeyPointContainer::Init(mpCamera->bDistorted, mpCamera->K, mpCamera->D);
			ObjPointContainer::Reset(N);
			BaseSLAM::StereoDataContainer::Init(N);
			/*UndistortKeyDatas(mpCamera->K, mpCamera->D);*/

			/*bool bDepth = mpCamera->mCamSensor == BaseSLAM::CameraSensor::RGBD;
			if (bDepth) {
				BaseSLAM::StereoDataContainer::ComputeStereoFromRGBD(mpCa, mbf, this);
			}*/
		}
		/*void AddData(cv::KeyPoint kp, const cv::Mat& d) {

			BaseSLAM::KeyPointContainer::AddData(kp, d);
			mvKeyDatas.push_back(kp);
			mDescriptors.push_back(d.clone());
		}*/
		cv::Mat GetCenter();
		cv::Mat GetPose();
		void SetPose(const cv::Mat& _T);

	public:
		//void Update() override {};
		
		bool UnprojectStereo(int i, cv::Mat& Xw, cv::Mat& Xo, const cv::Mat& Rwc, const cv::Mat& twc);
	public:
		std::vector<int> mvIDXs;
		std::map<int, int> mapIDXs;
		//BaseSLAM::DataContainer<cv::KeyPoint> KDC;
		//BaseSLAM::MapDataContainer<EdgeSLAM::MapPoint*> MDC;
		std::vector<EdgeSLAM::MapPoint*> mvpMapPoints;
		std::vector<cv::Mat> mvWorld, mvObject;
		float fx, fy, cx, cy, invfx, invfy;
		cv::Mat mUsed;
		cv::Mat origin;
		int n1, n2;
	public:
		BaseSLAM::AbstractCamera* mpCamera;
		BaseSLAM::AbstractPose* mpWorldPose;
		static std::atomic<int> nBoundingBoxId;
		int mnLabel;
		float mfConfidence;
		cv::Rect mRect;
		BoxFrame* mpRef;
		ObjectMap* mpMap;

		ObjectPointGraph* mpGraph;
	};
}
#endif