#ifndef OBJECT_SLAM_GAUSSIAN_OBJECT_H
#define OBJECT_SLAM_GAUSSIAN_OBJECT_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentSet.h>
#include <ConcurrentVector.h>

namespace ObjectSLAM {
	/*namespace GOMAP {
		class GaussianObject;
	}*/
	class FrameInstance;
	class InstanceMask;

	namespace GOMAP {
		
		class GO2D {
		public:
			GO2D(){}
			GO2D(const cv::Mat& _c, const cv::Mat& _cov, float _major, float _minor, float _angle)
				:center(_c), cov2D(_cov), major(_major), minor(_minor), angle_rad(_angle)
				, angle_deg(angle_rad *180.0/CV_PI), axes(cvRound(major), cvRound(minor))
			{
			}
			virtual ~GO2D(){}
		public:
			cv::Mat center;
			cv::Mat cov2D;
			cv::Rect rect;
			float major, minor, angle_rad, angle_deg;
			cv::Size axes; //angle_deg와 cv::ellipse가능

			cv::RotatedRect CalcEllipse(float chisq = 1.0);
			cv::Rect CalcRect(float chisq = 1.0);

			float CalcIOU(cv::Rect other) {
				cv::RotatedRect a;
				
				cv::Rect intersection = this->rect & other;
				if (intersection.empty()) return 0.0;

				float intersection_area = intersection.area();
				float union_area = this->rect.area() + other.area() - intersection_area;

				return intersection_area / union_area;
			}

			float CalcIOU(GO2D& other) {
				return CalcIOU(other.rect);
			}
		};
		class GaussianObject {
		public:
			GaussianObject() {}
			GaussianObject(const cv::Mat& _pos, const cv::Mat& _cov, const cv::Mat& _R);
			/*GaussianObject(const cv::Mat& pos, const cv::Mat& cov,
				const cv::Mat& feat, const cv::Rect2d& box,
				double obsNoise = 0.1);*/
			virtual ~GaussianObject() {}
		public:
			static std::atomic<long unsigned int> mnNextId;
			int id;
			
			cv::Mat Rwo; //3x3 행렬. 오브젝트에서 슬램 좌표계로 변환하는 매트릭스. 다이나믹에서는 고정하면 안됨.

			std::atomic<int> nObs;
			std::atomic<int> nSeg; //seg에서 바로 연결될 확률
			std::atomic<int> nContour;//전체 포인트의 수

			ConcurrentMap<InstanceMask*, FrameInstance*> mObservations;

			GO2D Project2D(const cv::Mat& K, const cv::Mat& Rcw, const cv::Mat& tcw);
			void AddObservation(InstanceMask* f, FrameInstance* obs, bool btype = true); //true이면 seg, false이면 sam
			FrameInstance* GetObservation(InstanceMask* f);
			std::map<InstanceMask*, FrameInstance*> GetObservations();

			float CalcDistance3D(GaussianObject* other);

			/*cv::Mat features;
			cv::Rect2d bbox;
			double observationNoise;*/
		public:
			void SetPosition(const cv::Mat& _pos);
			cv::Mat GetPosition();
			void SetCovariance(const cv::Mat& _cov);
			cv::Mat GetCovariance();
		private:
			std::mutex mMutex;
			cv::Mat mean;
			cv::Mat covariance;
			
		};
	}
}
#endif  