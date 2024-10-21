#ifndef OBJECT_SLAM_OBJECT_MATCHER_H
#define OBJECT_SLAM_OBJECT_MATCHER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

#include <BoxFrame.h>
#include <BoundingBox.h>

namespace EdgeSLAM {
	class Frame;
	class KeyFrame;
}

namespace ObjectSLAM {
	class ObjectMatchingInfo;
	class ObjectMatcher {
	public:
		static const int HISTO_LENGTH;
		static int SearchInstance(EdgeSLAM::Frame* pTarget, BoxFrame* pRef, const cv::Mat& gray1, const cv::Mat& gray2, ObjectMatchingInfo* pMatches);
		static int SearchByOpticalFlow(EdgeSLAM::Frame* pTarget, EdgeSLAM::KeyFrame* pRef, const cv::Mat& gray1, const cv::Mat& gray2, ObjectMatchingInfo* pMatches);
		static int SearchByOpticalFlow(EdgeSLAM::KeyFrame* pTarget, EdgeSLAM::KeyFrame* pRef, const cv::Mat& gray1, const cv::Mat& gray2, ObjectMatchingInfo* pMatches);

		static int SearchFrameAndFrame(BoxFrame* pF1, BoxFrame* pF2, std::vector<std::pair<int, int>>& vecMatches, const float thRadius = 50.0, const int thdist = 50, const float thratio = 0.9, bool bCheckOri = true);
		static int SearchBoxAndBox(BoundingBox* pB1, BoundingBox* pB2, std::vector<std::pair<int, int>>& vecMatches, const float thRadius = 15.0, const int thdist = 50, const float thratio = 0.9, bool bCheckOri = true);
		static int SearchFrameByProjection(BoundingBox* pBox, BoxFrame* pFrame, std::vector<std::pair<int, int>>& vecMatches, float thProjection = 3.0, float thMaxDesc = 100.0, float thMinDesc = 50.0, bool bCheckOri = false);
	};
}
#endif   m       v  