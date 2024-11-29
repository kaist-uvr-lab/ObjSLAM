#ifndef OBJECT_SLAM_GLOBAL_INSTANCE_H
#define OBJECT_SLAM_GLOBAL_INSTANCE_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

#include <MapPoint.h>
#include <ObjectSLAMTypes.h>

namespace EdgeSLAM {
	class MapPoint;
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
	};
}

#endif