#include <ObjectPointGraph.h>

#include <BoundingBox.h>
#include <MapPoint.h>
#include <ObjectPoint.h>

namespace ObjectSLAM {
	void ObjectPointGraph::SetBadFlag() {
		std::unique_lock<std::mutex> lock(mMutexConnections);
		//std::unique_lock<std::mutex> lock1(mMutexFeatures);

		for (auto pair : mConnectedKeyFrameWeights) {
			auto kf = pair.first;
			EraseConnection(kf);
		}

		mConnectedKeyFrameWeights.clear();
		mvpOrderedConnectedKeyFrames.clear();

		// Update Spanning Tree
		std::set<BoundingBox*> sParentCandidates;
		sParentCandidates.insert(mpParent);

		while (!mspChildrens.empty())
		{
			bool bContinue = false;

			int max = -1;
			BoundingBox* pC = nullptr;
			BoundingBox* pP = nullptr;

			for (auto pKF : mspChildrens) {
				if (pKF->isBad())
					continue;
				auto gKF = pKF->mpGraph;
				// Check if a parent candidate is connected to the keyframe
				std::vector<BoundingBox*> vpConnected = gKF->GetVectorCovisibleFrames();
				for (size_t i = 0, iend = vpConnected.size(); i < iend; i++)
				{
					auto pV = vpConnected[i];
					auto g1 = pV->mpGraph;
					for (auto pP2 : sParentCandidates) {

						if (pV->mnId == pP2->mnId)
						{
							auto g2 = pP2->mpGraph;
							int w = gKF->GetWeight(pV);
							if (w > max) {
								pC = pKF;
								pP = pV;
								max = w;
								bContinue = true;
							}
						}
					}
				}
			}

			if (bContinue && pC && pP)
			{
				auto tG = pC->mpGraph;
				tG->ChangeParent(pP);
				sParentCandidates.insert(pC);
				mspChildrens.erase(pC);
			}
			else
				break;
		}

		if (!mspChildrens.empty())
			for (auto pKF : mspChildrens) {
				auto tG = pKF->mpGraph;
				tG->ChangeParent(mpParent);
			}
		auto gParent = mpParent->mpGraph;
		gParent->EraseChild((BoundingBox*)this);
	}
	void ObjectPointGraph::UpdateConnections(int th) {
		std::map<BoundingBox*, int> KFcounter;
		BoundingBox* pThis = (BoundingBox*)this;
		auto vecMapDatas = pThis->mvpMapDatas.get();

		for (auto vit = vecMapDatas.begin(), vend = vecMapDatas.end(); vit != vend; vit++)
		{
			auto pOP = *vit;
			if (!pOP)
				continue;

			auto observations = pOP->mpObservation->GetObservations();

			for (auto mit = observations.begin(), mend = observations.end(); mit != mend; mit++)
			{
				if (mit->first->mpGraph == this)
					continue;
				KFcounter[mit->first]++;
			}

		}

		if (KFcounter.empty())
			return;

		int nmax = 0;
		BoundingBox* pKFmax = nullptr;

		std::vector<std::pair<int, BoundingBox*> > vPairs;
		//vPairs.reserve(KFcounter.size());

		for (auto mit = KFcounter.begin(), mend = KFcounter.end(); mit != mend; mit++)
		{

			if (mit->second > nmax)
			{
				nmax = mit->second;
				pKFmax = mit->first;
			}
			if (mit->second >= th)
			{
				vPairs.push_back(std::make_pair(mit->second, mit->first));
				(mit->first)->mpGraph->AddConnection(pThis, mit->second);
			}
		}

		if (vPairs.empty())
		{
			vPairs.push_back(std::make_pair(nmax, pKFmax));
			pKFmax->mpGraph->AddConnection(pThis, nmax);
		}

		sort(vPairs.begin(), vPairs.end());
		std::list<BoundingBox*> lKFs;
		std::list<int> lWs;
		for (size_t i = 0; i < vPairs.size(); i++)
		{
			lKFs.push_front(vPairs[i].second);
			lWs.push_front(vPairs[i].first);
		}

		{
			std::unique_lock<std::mutex> lock(mMutexConnections);

			// mspConnectedKeyFrames = spConnectedKeyFrames;
			mConnectedKeyFrameWeights = KFcounter;
			mvpOrderedConnectedKeyFrames = std::vector<BoundingBox*>(lKFs.begin(), lKFs.end());
			mvOrderedWeights = std::vector<int>(lWs.begin(), lWs.end());

			//if (mbFirstConnection && mnId != 0)
			{
				mpParent = mvpOrderedConnectedKeyFrames.front();
				mpParent->mpGraph->AddChild(pThis);
				mbFirstConnection = false;
			}

		}

	}
	void ObjectPointGraph::UpdateLocalMap(std::vector<BoundingBox*>& vpLocalKFs, std::vector<ObjectPoint*>& vpLocalMapDatas) {
		UpdateKeyFrames(vpLocalKFs);
		UpdateLocalMapDatas(vpLocalKFs, vpLocalMapDatas);
	}
	void ObjectPointGraph::UpdateKeyFrames(std::vector<BoundingBox*>& vpLocalKFs) {
		std::unordered_map<BoundingBox*, int> keyframeCounter;
		BoundingBox* targetSF = mpBox;
		auto mvpMapLines = targetSF->mvpMapDatas.get();
		int N = mvpMapLines.size();
		for (int i = 0; i < N; i++)
		{
			auto pML = mvpMapLines[i];
			if (pML && !pML->isBad()) {
				const std::map<BoundingBox*, size_t> observations = pML->mpObservation->GetObservations();
				for (std::map<BoundingBox*, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
					keyframeCounter[it->first]++;
			}
			else {
				targetSF->mvpMapDatas.update(i, nullptr);
			}
		}

		if (keyframeCounter.empty())
			return;
		std::set< BoundingBox*> mspLocalKFs;
		int max = 0;
		BoundingBox* pKFmax = static_cast<BoundingBox*>(nullptr);

		// All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
		for (auto it : keyframeCounter) {
			BoundingBox* pKF = it.first;

			/*if (pKF->isBad())
				continue;*/

			if (it.second > max)
			{
				max = it.second;
				pKFmax = pKF;
			}

			vpLocalKFs.push_back(it.first);
			mspLocalKFs.insert(pKF);
		}

		//// Include also some not-already-included keyframes that are neighbors to already-included keyframes
		////for (std::vector<KeyFrame*>::const_iterator itKF = vpLocalKFs.begin(), itEndKF = vpLocalKFs.end(); itKF != itEndKF; itKF++)
		for (size_t i = 0, iend = vpLocalKFs.size(); i < iend; i++)
		{
			// Limit the number of keyframes
			if (vpLocalKFs.size() > 80)
				break;

			BoundingBox* pKF = vpLocalKFs[i];// *itKF;

			const std::vector<BoundingBox*> vNeighs = pKF->mpGraph->GetBestCovisibilityFrames(10);

			for (std::vector<BoundingBox*>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
			{
				BoundingBox* pNeighKF = *itNeighKF;
				if (pNeighKF && !pNeighKF->isBad() && !mspLocalKFs.count(pNeighKF))
				{
					mspLocalKFs.insert(pNeighKF);
					break;
				}
			}

			const std::set<BoundingBox*> spChilds = pKF->mpGraph->GetChilds();
			for (std::set<BoundingBox*>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
			{
				BoundingBox* pChildKF = *sit;
				if (pChildKF && !pChildKF->isBad() && !mspLocalKFs.count(pChildKF))
				{
					mspLocalKFs.insert(pChildKF);
					break;
				}
			}

			BoundingBox* pParent = pKF->mpGraph->GetParent();
			if (pParent && !pParent->isBad() && !mspLocalKFs.count(pParent))
			{
				mspLocalKFs.insert(pParent);
				break;
			}

		}
	}
	void ObjectPointGraph::UpdateLocalMapDatas(const std::vector<BoundingBox*>& vpLocalKFs, std::vector<ObjectPoint*>& vpLocalMapDatas) {
		std::set<ObjectPoint*> spMPs;
		for (auto pKF : vpLocalKFs) {
			const std::vector<ObjectPoint*> vpMPs = pKF->mvpMapDatas.get();
			for (auto pMP : vpMPs) {
				if (!pMP || pMP->isBad() || spMPs.count(pMP))
					continue;
				vpLocalMapDatas.push_back(pMP);
				spMPs.insert(pMP);
			}
		}
	}
}