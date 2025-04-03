#ifndef OBJECT_SLAM_GLOBAL_INSTANCE_H
#define OBJECT_SLAM_GLOBAL_INSTANCE_H
#pragma once

//SQ-SLAM의 객체를 생성하는 코드로

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

#include <MapPoint.h>
#include <ObjectSLAMTypes.h>

namespace EdgeSLAM {
	class MapPoint;
	class KeyFrame;
}

namespace ObjectSLAM {

	class ObjectSLAM;
	class BoxFrame;
	class FrameInstance;

	class GlobalInstance {
	public:
		GlobalInstance();
		virtual ~GlobalInstance() {

		}
	public:
		void Merge(GlobalInstance* pG);
		/*void Connect(BoxFrame* pBF, int id) {
			mapConnected.Update(pBF, id);
		}*/
		void EIFFilterOutlier();
		//키프레임 연결. spherical coordinate으로 관리
		void Update(EdgeSLAM::KeyFrame* pKF);
		//spherical coordinate으로 현재 프레임과 관련된 맵포인트 획득
		void GetLocalMPs(std::set<EdgeSLAM::MapPoint*>& spMPs,EdgeSLAM::KeyFrame* pKF, float angle, float dist, int bin = 1);
		//local mp가 현재 프레임에 있는지 프로젝션
		void GetProjPTs(const std::set<EdgeSLAM::MapPoint*>& spMPs, std::vector<cv::Point2f>& vecPTs, EdgeSLAM::KeyFrame* pKF);

		cv::Point2f GetCenter(const std::vector<cv::Point2f>& points);
		cv::Rect GetRect(const std::vector<cv::Point2f>& points);

		void Connect(FrameInstance* pIns, BoxFrame* pBF, int id);
		//mask의 맵포인트 추가
		void AddMapPoints(std::set<EdgeSLAM::MapPoint*> spMPs);

		////Position
	public:
		//isBad
		//Merge
		static ObjectSLAM* ObjSystem;
		std::atomic<bool> mbBad;
		//bool isOutlier(const cv::Rect& rect, cv::Point2f pt);
		void RemoveOutlier();
		void Update(std::vector<cv::Mat>& mat, float val = 1.285);
		cv::Point2f ProjectPoint(const cv::Mat T, const cv::Mat& K);
		cv::Mat GetPosition();
		void UpdatePosition();
		//void UpdatePosition(std::vector<)
	private:
		std::mutex mMutexPos;
		cv::Mat pos;
		cv::Mat axis;
		float sx, sy, sz;
		////Position 
		////3D Bounding Box
	public:
		void CalculateBoundingBox();
		void ProjectBB(std::vector<cv::Point2f>& vecProjPoints, const cv::Mat& K, const cv::Mat& T);
		void DrawBB(cv::Mat& image, const std::vector<cv::Point2f>& projectedCorners);
	private:
		std::mutex mMutexBB;
		std::vector<cv::Point3f> vecCorners;


		////3D Bounding Box
	public:
		int mnId;
		std::atomic<int> mnMatchFail;
		ConcurrentSet<EdgeSLAM::MapPoint*> AllMapPoints;
		ConcurrentMap<BoxFrame*, int> mapConnected;
		ConcurrentMap<FrameInstance*, int> mapInstances;
		static std::atomic<long unsigned int> mnNextGIId;
		ConcurrentMap<std::pair<int,int>, std::set<EdgeSLAM::KeyFrame*>> MapKFs;
		ConcurrentMap<std::pair<int, int>, std::map<int, std::set<EdgeSLAM::MapPoint*>>> MapMPs;
	};
}

#endif