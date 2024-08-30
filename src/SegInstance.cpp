#include <SegInstance.h>
#include <BoxFrame.h>
#include <KeyPointContainer.h>
#include <StereoDataContainer.h>
#include <ObjectPointGraph.h>

namespace ObjectSLAM {
	std::atomic<int> SegInstance::nSegInstanceId = 0;
	SegInstance::SegInstance(BoxFrame* _ref, int _fx, int _fy, int _cx, int _cy, int _label, float _conf, bool _thing, BaseSLAM::BaseDevice* Device)
		:BaseSLAM::AbstractFrame(Device, 0), BaseSLAM::KeyPointContainer(Device->mpCamera), BaseSLAM::StereoDataContainer(),
		ObjPointContainer(),
		mpRef(_ref), mpMap(nullptr), n1(0), n2(0), origin(cv::Mat::zeros(3, 1, CV_32FC1)),mbIsthing(_thing),
		mpWorldPose(new BaseSLAM::AbstractPose()), fx(_fx), fy(_fy), cx(_cx), cy(_cy), invfx(1.0 / fx), invfy(1.0 / fy), mnConnected(0)
	{
		mnId = (++nSegInstanceId);
		mpCamera = Device->mpCamera;
		mnLabel = _label;
		mfConfidence = _conf;
		//mRect = cv::Rect(left, right);
		//mUsed = cv::Mat::zeros(mRect.height, mRect.width, CV_8UC1);
	}

	cv::Mat SegInstance::GetCenter() {
		return mpRef->GetCameraCenter();
	}
	cv::Mat SegInstance::GetPose() {
		return mpWorldPose->GetPose();
	}
	void SegInstance::SetPose(const cv::Mat& _T) {
		mpWorldPose->SetPose(_T);
	}

	bool SegInstance::UnprojectStereo(int i, cv::Mat& Xw, cv::Mat& Xo, const cv::Mat& Rwc, const cv::Mat& twc) {
		
	}

	void SegInstance::UpdateInstance(SegInstance* pConnected) {
		auto allIns = pConnected->mConnectedInstances.ConvertVector();
		allIns.push_back(pConnected);
		for (auto pIns : allIns) {
			if (pIns == this)
				continue;
			if (!mConnectedInstances.Count(pIns)) {
				mConnectedInstances.Update(pIns);
				mnConnected++;
			}
		}
	}
}

