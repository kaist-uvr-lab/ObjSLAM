#ifndef OBJECT_SLAM_DEVICE_H
#define OBJECT_SLAM_DEVICE_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

namespace ObjectSLAM {
	class BoxFrame;
	class ObjectDevice {
	public:
		ObjectDevice(): mpPrevBF(nullptr), mpCurrBF(nullptr){}
		virtual~ ObjectDevice(){}
	public:
		BoxFrame* mpPrevBF, *mpCurrBF;
		//cv::Mat imPrev, imPrevGray, imCurr, imCurrGray;
	private:

	};
}
#endif