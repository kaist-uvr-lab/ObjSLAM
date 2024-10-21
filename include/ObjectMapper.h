#ifndef OBJECT_SLAM_OBJECT_MAPPER_H
#define OBJECT_SLAM_OBJECT_MAPPER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

#include <AbstractPose.h>

namespace EdgeSLAM {
	class KeyFrame;
	class Frame;
}

namespace ObjectSLAM{
	class BoundingBox;
	class BoxFrame;
	class ObjectSLAM;
	class SegInstance;

	class ObjectMapper {
	public:
		static int TwoViewTriangulation(EdgeSLAM::Frame* pB1, EdgeSLAM::KeyFrame* pB2, const std::vector<std::pair<int, int>>& vMatches, std::vector<std::pair<bool, cv::Mat>>& vecTriangulated);
		static int TwoViewTriangulation(EdgeSLAM::KeyFrame* pB1, EdgeSLAM::KeyFrame* pB2, const std::vector<std::pair<int, int>>& vMatches, std::vector<std::pair<bool, cv::Mat>>& vecTriangulated);

		static int TwoViewTriangulation(SegInstance* pB1, SegInstance* pB2, const std::vector<std::pair<int, int>>& vMatches, std::vector<std::pair<bool, cv::Mat>>& vecTriangulated);
		static int TwoViewTriangulation(BoundingBox* pB1, BoundingBox* pB2, const std::vector<std::pair<int, int>>& vMatches, std::vector<std::pair<bool, cv::Mat>>& vecTriangulated);

		static void ObjectMatchingPropagation(ObjectSLAM* System, const std::string& src, int id, EdgeSLAM::KeyFrame* pKF, BoxFrame* pBF);
	}; 
}
#endif