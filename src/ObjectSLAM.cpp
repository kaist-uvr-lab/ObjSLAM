#include <ObjectSLAM.h>

#include <KeyFrame.h>

namespace ObjectSLAM {
	std::vector<BoxFrame*> ObjectSLAM::GetConnectedBoxFrames(EdgeSLAM::KeyFrame* pKF, int nn) {
		std::vector<BoxFrame*> res;
		auto vecKFs = pKF->GetBestCovisibilityKeyFrames(nn);
		for (auto pKF : vecKFs) {
			if (!MapKeyFrameNBoxFrame.Count(pKF))
				continue;
			auto pBF = MapKeyFrameNBoxFrame.Get(pKF);
			res.push_back(pBF);
		}
		return res;
	}
}