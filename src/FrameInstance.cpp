#include <FrameInstance.h>

#include <KeyFrame.h>
#include <MapPoint.h>

namespace ObjectSLAM {

	void FrameInstance::Update(EdgeSLAM::KeyFrame* pKF) {
		if (contour.size() == 0)
			return;

		auto vecMPs = pKF->mvpMapPoints.get();
		for (int i = 0; i < pKF->N; i++)
		{
			auto pt = pKF->mvKeys[i].pt;
			if (cv::pointPolygonTest(contour, pt, false) < 0.0)
				continue;
			//kp¿Í descriptorÃß°¡
			mvKeys.push_back(pKF->mvKeys[i]);
			mDescriptor.push_back(pKF->mDescriptors.row(i));
			
			auto pMPi = vecMPs[i];
			this->setKPs.insert(i);
			if (pMPi && !pMPi->isBad())
			{
				this->setMPs.insert(pMPi);
			}
		}
	}

	FrameInstance* FrameInstance::ConvertedInstasnce(EdgeSLAM::KeyFrame* pKF, cv::Point2f apt)
	{
		auto pCurr = new FrameInstance(pKF);
		pCurr->pt = this->pt + apt;
		pCurr->rect = this->rect;
		pCurr->rect.x += apt.x;
		pCurr->rect.y += apt.y;
		
		for (auto pt : this->contour) {
			auto npt = pt;
			npt.x += apt.x;
			npt.y += apt.y;
			pCurr->contour.push_back(npt);
		}

		pCurr->area = this->area;

		//mask
		std::vector<std::vector<cv::Point>> contours;
		contours.push_back(pCurr->contour);
		cv::Mat newmask = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC1);;
		cv::drawContours(newmask, contours, 0, cv::Scalar(255, 255, 255), -1);
		pCurr->mask = newmask.clone();
		return pCurr;
	}

}