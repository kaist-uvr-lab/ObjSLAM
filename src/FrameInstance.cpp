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
			auto pMPi = vecMPs[i];

			this->setKPs.insert(i);
			if (pMPi && !pMPi->isBad())
			{
				this->setMPs.insert(pMPi);
			}
		}
	}

}