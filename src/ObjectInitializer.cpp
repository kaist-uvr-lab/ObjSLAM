#include <ObjectInitializer.h>
#include <BoundingBox.h>
#include <ObjectMap.h>

namespace ObjectSLAM {
	//bool ObjectInitializer::StereoInitialization(BoundingBox* pBox, ObjectMap* pObjMap) {
	ObjectMap* ObjectInitializer::StereoInitialization(BoundingBox* pBox) {
		if (pBox->N < 20){
			return nullptr;
			//return false;
		}
		cv::Mat Rwc = pBox->mpWorldPose->GetPose().rowRange(0,3).colRange(0,3).t();
		cv::Mat twc = pBox->mpWorldPose->GetCameraCenter();

		pBox->mvWorld = std::vector<cv::Mat>(pBox->N, cv::Mat());
		pBox->mvObject = std::vector<cv::Mat>(pBox->N, cv::Mat());

		float Nres = 0;
		cv::Mat R = cv::Mat::eye(3, 3, CV_32FC1);
		cv::Mat t = cv::Mat::zeros(3, 1, CV_32FC1);

		for (int i = 0, N = pBox->N; i < N; i++) {
			cv::Mat Xw, Xo;
			bool bRes = pBox->UnprojectStereo(i, Xw, Xo, Rwc, twc);
			if (bRes) {
				pBox->mvWorld[i] = Xw.clone();
				pBox->mvObject[i] = Xo.clone();
				t += Xw;
				Nres++;
			}
		}
		
		if (Nres < 20) {
			return nullptr;
			//return false;
		}
		t /= Nres;

		auto pObjMap = new ObjectMap();
		cv::Mat T = cv::Mat::eye(4, 4, CV_32FC1);
		R.copyTo(T.rowRange(0, 3).colRange(0, 3));
		t.copyTo(T.rowRange(0, 3).col(3));
		pObjMap->mpWorldPose->SetPose(T);
		pObjMap->vecBoundingBoxes.push_back(pBox) ;
		return pObjMap;
		//return true;

	}
}