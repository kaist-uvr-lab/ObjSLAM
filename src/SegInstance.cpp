#include <SegInstance.h>
#include <BoxFrame.h>
#include <KeyPointContainer.h>
#include <StereoDataContainer.h>
#include <ObjectPointGraph.h>
#include <SemanticLabel.h>;

namespace ObjectSLAM {
	std::atomic<int> SegInstance::nSegInstanceId = 0;
	SegInstance::SegInstance(BoxFrame* _ref, int _fx, int _fy, int _cx, int _cy, int _label, float _conf, bool _thing, BaseSLAM::BaseDevice* Device, bool _bdetected)
		:BaseSLAM::AbstractFrame(Device, 0), BaseSLAM::KeyPointContainer(Device->mpCamera), BaseSLAM::StereoDataContainer(),
		ObjPointContainer(), mbDetected(_bdetected), 
		mpRef(_ref), mpMap(nullptr), n1(0), n2(0), origin(cv::Mat::zeros(3, 1, CV_32FC1)),mbIsthing(_thing),
		mpWorldPose(new BaseSLAM::AbstractPose()), fx(_fx), fy(_fy), cx(_cx), cy(_cy), invfx(1.0 / fx), invfy(1.0 / fy), mnConnected(0)
	{
		mnId = (++nSegInstanceId);
		mpCamera = Device->mpCamera;
		//mnLabel = _label;
		//mfConfidence = _conf;

		/*if (isTable(_label, mbIsthing) || isFloor(_label, mbIsthing))
			_conf *= 0.1;*/

		if (isTable(_label, mbIsthing)) {
			_conf *= 0.1;
			_label = 42;
			mbIsthing = false;
		}
		if (isFloor(_label, mbIsthing)) {
			_conf *= 0.1;
			_label = 8;
		}

		mpConfLabel = new EdgeSLAM::SemanticConfLabel();
		mpConfLabel->Update(_label, _conf, mbIsthing);

		/*auto pconf = new EdgeSLAM::SemanticConfidence(mbIsthing);
		pconf->Add(mfConfidence);
		mpConfLabel->LabelConfCount.Update(mnLabel, pconf);
		mpConfLabel->label = mnLabel;
		mpConfLabel->maxConf = mfConfidence;*/
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
				mpConfLabel->Update(pIns->mpConfLabel->label, pIns->mpConfLabel->maxConf, pIns->mbIsthing);
			}
		}
	}
	bool SegInstance::isTable(int _label, bool _thing) {
		return (_label == 60 && _thing) || (_label == 42 && !_thing);
	}
	bool SegInstance::isFloor(int _label, bool _thing){
		//44 pavement
		return (_label == 8 || _label == 43 || _label == 44) && !_thing;
	}
	bool SegInstance::isTable() {
		return (mpConfLabel->label == 60 && mbIsthing) || (mpConfLabel->label == 42 && !mbIsthing);
	}
	bool SegInstance::isFloor() {
		return (mpConfLabel->label == 8 || mpConfLabel->label == 43 || mpConfLabel->label == 44) && !mbIsthing;
	}
	bool SegInstance::isCeiling() {
		return (mpConfLabel->label == 39) && !mbIsthing;
	}
	bool isWall() {
		//30, 31, 32, 33, 52
		//return (label == 8 || label == 43) && !bIsThing;
		return false;
	}

	bool SegInstance::isObject() {
		return mbIsthing || (!mbIsthing && mpConfLabel->label == 0);
	}

	bool SegInstance::isStaticObject() {
		return (isTable() || isFloor() || isCeiling());
	}

}

