#include <ObjectSLAM.h>

#include <KeyFrame.h>
#include <MapPoint.h>
#include <SemanticLabel.h>
#include <SegInstance.h>
#include <BoxFrame.h>

namespace ObjectSLAM {

	void ObjectSLAM::SaveObjectAsso() {
		std::stringstream ss;
		ss << "../res/aaresasso/res.csv";

		//file open
		std::ofstream file;
		file.open(ss.str(), std::ios::trunc);

		//iou
		ss.str("");
		auto vec = VecIOU.get();
		float avgIOU = 0.0;
		for (auto val : vec)
		{
			avgIOU += val;
		}
		avgIOU /= vec.size();
		{
			std::stringstream ss;
			ss << "AVG IOU," << avgIOU << std::endl;
		}
		//iou

		//detection
		auto mapDatas = MapNumObjects.Get();
		for (auto pair : mapDatas)
		{
			auto id = pair.first;
			auto data = pair.second;
			ss << data << std::endl;
		}
		//detection
		
		//log
		for (auto strres : vecObjectAssoRes) {
			ss << strres << std::endl;
		}
		//log
		
		file.write(ss.str().c_str(), ss.str().size());
		file.close();
	}

	void ObjectSLAM::SaveLatency(std::string keyword)
	{
		std::cout << "save " << keyword << std::endl;
		
		auto vecDatas = MapLatency.Get(keyword);
		std::stringstream ss;
		ss << "../res/aseglatency/" << keyword << ".csv";

		std::ofstream file;
		file.open(ss.str(), std::ios::app);
		ss.str("");
		for (int i = 0, N = vecDatas.size(); i < N; i++) {
			ss << vecDatas[i]<<std::endl;
		}
		file.write(ss.str().c_str(), ss.str().size());
		file.close();
		
		MapLatency.Update(keyword, std::vector<double>());
		std::cout << "save " << keyword << "end"<<std::endl;
	}
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
				if (idx < 0)
					continue;
				auto tempConfLabel = pBFi->mvpConfLabels[idx];
				if (!tempConfLabel)
					continue;
				 
				auto sid = pBFi->mvnInsIDs[idx];
				if (sid < 0)
					continue;
				auto pIns = pBFi->mmpBBs[sid];
				//auto tempConfLabel = pIns->mpConfLabel;
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