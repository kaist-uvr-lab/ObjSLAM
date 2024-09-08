#include <ObjectSLAM.h>

#include <KeyFrame.h>
#include <MapPoint.h>
#include <SemanticLabel.h>
#include <SegInstance.h>
#include <BoxFrame.h>

namespace ObjectSLAM {

	void ObjectSLAM::UpdateMapPoint(BoxFrame* pF) {
		auto pKF = pF->mpRefKF;
		for (int i = 0; i < pKF->N; i++) {
			auto pMPi = pKF->mvpMapPoints.get(i);
			if (!pMPi || pMPi->isBad())
				continue;
			auto obs = pMPi->GetObservations();
			auto pConfLabel = new EdgeSLAM::SemanticConfLabel();
			for (auto pair : obs) {
				auto pKFi = pair.first;
				auto idx = pair.second;
				
				if (!MapKeyFrameNBoxFrame.Count(pKFi->mnFrameId))
				{
					//std::cout << "err::update mp = " << pKFi->mnFrameId <<" "<<idx << std::endl;
					continue;
				}
				auto pBFi = MapKeyFrameNBoxFrame.Get(pKFi->mnFrameId);
				auto sid = pBFi->mvnInsIDs[idx];
				if (sid < 0)
					continue;

				auto pIns = pBFi->mmpBBs[sid];
				
				auto tempConfLabel = pIns->mpConfLabel;

				pConfLabel->Update(tempConfLabel->label, tempConfLabel->maxConf, pIns->mbIsthing);
			}
			pMPi->mpConfLabel->label = pConfLabel->label.load();
			pMPi->mpConfLabel->maxConf = pConfLabel->maxConf.load();
		}
	}

	std::vector<BoxFrame*> ObjectSLAM::GetConnectedBoxFrames(EdgeSLAM::KeyFrame* _pKF, int nn) {
		std::vector<BoxFrame*> res;
		auto vecKFs = _pKF->GetBestCovisibilityKeyFrames(nn);
		for (auto pKF : vecKFs) {
			if (!MapKeyFrameNBoxFrame.Count(pKF->mnFrameId))
				continue;
			auto pBF = MapKeyFrameNBoxFrame.Get(pKF->mnFrameId);
			if(pBF)
				res.push_back(pBF);
		}
		return res;
	}
}