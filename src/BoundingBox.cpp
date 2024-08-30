#include <BoundingBox.h>
#include <BoxFrame.h>
#include <KeyPointContainer.h>
#include <StereoDataContainer.h>
#include <ObjectPointGraph.h>

namespace ObjectSLAM {
	std::atomic<int> BoundingBox::nBoundingBoxId = 0;
	//instance
	BoundingBox::BoundingBox(BoxFrame* _ref, int _fx, int _fy, int _cx, int _cy, int _label, float _conf, cv::Point2f left, cv::Point2f right, BaseSLAM::BaseDevice* Device)
		:BaseSLAM::AbstractFrame(Device, 0), BaseSLAM::KeyPointContainer(Device->mpCamera), BaseSLAM::StereoDataContainer(), 
		ObjPointContainer(),
		mpRef(_ref), mpMap(nullptr), n1(0), n2(0), origin(cv::Mat::zeros(3,1,CV_32FC1)),
		mpWorldPose(new BaseSLAM::AbstractPose()),fx(_fx), fy(_fy), cx(_cx), cy(_cy), invfx(1.0/fx), invfy(1.0/fy)
		//, KDC(BaseSLAM::DataContainer<cv::KeyPoint>()), MDC(BaseSLAM::MapDataContainer<EdgeSLAM::MapPoint*>()) 
	{
		mnId = (++nBoundingBoxId);
		mpCamera = Device->mpCamera;
		mnLabel = _label;
		mfConfidence = _conf;
		mRect = cv::Rect(left, right);
		mUsed = cv::Mat::zeros(mRect.height, mRect.width, CV_8UC1);
		mpGraph = new ObjectPointGraph();
		mpGraph->mpBox = this;
		//desc = cv::Mat::zeros(0, 32, CV_8UC1);
	}

	cv::Mat BoundingBox::GetCenter() {
		return mpRef->GetCameraCenter();
	}
	cv::Mat BoundingBox::GetPose() {
		return mpWorldPose->GetPose();
	}
	void BoundingBox::SetPose(const cv::Mat& _T) {
		mpWorldPose->SetPose(_T);
	}

	bool BoundingBox::UnprojectStereo(int i, cv::Mat& Xw, cv::Mat& Xo, const cv::Mat& Rwc, const cv::Mat& twc) {
		/*const float z = mvDepth[i];
		if (z > 0)
		{
			const float u = mvKeyDataUns[i].pt.x;
			const float v = mvKeyDataUns[i].pt.y;
			const float x = (u - cx) * z * invfx;
			const float y = (v - cy) * z * invfy;
			Xo = (cv::Mat_<float>(3, 1) << x, y, z);
			Xw = Rwc * Xo + twc;
			return true;
		}
		else
			return false;*/
	}
}