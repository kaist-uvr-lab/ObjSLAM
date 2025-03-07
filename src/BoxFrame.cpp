#include <BoxFrame.h>
#include <BoundingBox.h>
#include <SegInstance.h>
#include <KeyFrame.h>
#include <Frame.h>
#include <MapPoint.h>

#include <ObjectSLAM.h>
#include <FrameInstance.h>
#include <GlobalInstance.h>
#include <SemanticLabel.h>
#include <Utils_Geometry.h> 


namespace ObjectSLAM {

	bool isInRange(float val, float _min, float _max)
	{
		if (val > _min && val < _max)
			return true;
		return false;
	}

	
	ObjectSLAM* BoxFrame::ObjSystem = nullptr;
	
	BoxFrame::BoxFrame(int _id) :BaseSLAM::AbstractFrame(_id), mbInitialized(false), mpPrevBF(nullptr), mbYolo(false), mbDetectron2(false), mbSam2(false)
	{}
	BoxFrame::BoxFrame(int _id, const int w, const int h, BaseSLAM::BaseDevice* Device, BaseSLAM::AbstractPose* _Pose) : BaseSLAM::AbstractFrame(Device, _Pose, _id), BaseSLAM::KeyPointContainer(mpCamera), BaseSLAM::StereoDataContainer(), mUsed(cv::Mat::zeros(h, w, CV_32SC1)),
		mbYolo(false), mbDetectron2(false), mbSam2(false),
		mpKC(this), mpSC(this), mpRefKF(nullptr), mpDevice(Device), mbInitialized(false), mpPrevBF(nullptr)
	{
	}
	BoxFrame::~BoxFrame() {
		std::vector<BoundingBox*>().swap(mvpBBs);
		img.release();
		labeled.release();
		depth.release();
	}

	void BoxFrame::GetNeighGlobalInstnace(std::set<GlobalInstance*>& setGlobalIns) {
		if (!this->mapMasks.Count("yoloseg"))
			return;
		auto pCurrSeg = this->mapMasks.Get("yoloseg");
		
		auto pKF = this->mpRefKF;
		if (!pKF)
			return;

		auto pCurrSegInstances = pCurrSeg->FrameInstances.Get();
		auto vpNeighBFs = ObjSystem->GetConnectedBoxFrames(pKF, 10);
		
		for (auto pBF : vpNeighBFs) {

			if (pBF->mapMasks.Count("yoloseg"))
			{
				auto mapGlobals = pBF->mapMasks.Get("yoloseg")->MapInstances.Get();
				for (auto pair : mapGlobals) {
					auto pG = pair.second;
					
					if (!pG || pG->mbBad) {
						continue;
					}
					if (!setGlobalIns.count(pG))
						setGlobalIns.insert(pG);
				}
			}
		}
	}

	void BoxFrame:: UpdateInstanceKeyPoints(const std::vector<std::pair<cv::Point2f, cv::Point2f>>& vecPairPoints, const std::vector<std::pair<int, int>>& vecMatches, std::map < std::pair<int, int>, std::pair<int, int>>& mapChangedIns) {
		std::map<int, std::vector<cv::Point2f>> mapPoints;

		float w = mUsed.cols - 1;
		float h = mUsed.rows - 1;

		for (int i = 0; i < vecMatches.size(); i++) {
			auto pid = vecMatches[i].first;
			auto cid = vecMatches[i].second;

			auto pair = std::make_pair(pid, cid);

			if (!mapChangedIns.count(pair))
				continue;
			auto newPair = mapChangedIns[pair];

			//std::cout << "a" << std::endl;
			if (!mmpBBs.count(cid))
				std::cout << "err = old curr ins " << std::endl;
			if (!mmpBBs.count(newPair.second))
				std::cout << "err = new curr ins =" << mnId << " " << newPair.second << std::endl;
			auto pOldIns = mmpBBs[cid];
			auto pNewIns = mmpBBs[newPair.second];
			//std::cout << "b" << std::endl;
			
			auto pt = vecPairPoints[i].second;
			mapPoints[newPair.second].push_back(pt);
		}

		//mask filling
		for (auto pair : mapPoints) {
			auto sid = pair.first;
			auto vecPoints = pair.second;

			float min_x = 1000.0;
			float max_x = 0.0;
			float min_y = 1000.0;
			float max_y = 0.0;

			for (auto pt : vecPoints) {
				if (pt.x > max_x) {
					max_x = pt.x;
				}
				if (pt.x < min_x) {
					min_x = pt.x;
				}
				if (pt.y > max_y) {
					max_y = pt.y;
				}
				if (pt.y < min_y) {
					min_y = pt.y;
				}
			}
			min_x -= 2;
			min_y -= 2;
			max_x += 2;
			max_y += 2;
			if (min_x < 0.0)
				min_x = 0.0;
			if (max_x > w)
				max_x = w;
			if (min_y < 0.0)
				min_y = 0.0;
			if (max_y > h)
				max_y = h;
			auto rect = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
			{
				std::unique_lock<std::mutex> lock(mMutexInstance);
				cv::rectangle(seg, rect, cv::Scalar(sid, 0, 0), -1);
			}
			
			//seg(rect) = sid;
			//std::cout << "new ins rect = " << rect << std::endl;
		}
		//Ű����Ʈ üũ
		
		for (int i = 0; i < N; i++) {
			auto pt = mvKeyDatas[i].pt;
			int sid = GetInstance(pt);
			if (mapPoints.count(sid)) {
				mvnInsIDs[i] = sid;
			}
		}
	}
	void BoxFrame::UpdateInstanceKeyPoints(const std::vector<std::pair<int, int>>& vecMatches, const std::vector<int>& vecIDXs, std::map<std::pair<int,int>, std::pair<int, int>>& mapChangedIns) {
		
		std::map<int, std::vector<cv::Point2f>> mapPoints;

		float w = mUsed.cols-1;
		float h = mUsed.rows-1;

		for (int i = 0; i < vecMatches.size(); i++) {
			auto pid = vecMatches[i].first;
			auto cid = vecMatches[i].second;

			auto pair = std::make_pair(pid, cid);

			if (!mapChangedIns.count(pair))
				continue;
			auto newPair = mapChangedIns[pair];

			//std::cout << "a" << std::endl;
			if (!mmpBBs.count(pid))
				std::cout << "err = old ins " << std::endl;
			if (!mmpBBs.count(newPair.first))
				std::cout << "err = new ins =" <<mnId<<" " << newPair.first << std::endl;
			auto pOldIns = mmpBBs[pid];
			auto pNewIns = mmpBBs[newPair.first];
			//std::cout << "b" << std::endl;
			int idx = vecIDXs[i];
			
			mvnInsIDs[idx] = newPair.first;
			
			//int label = pNewIns->mnLabel;
			//mvpConfLabels[idx]->Update(pNewIns->mnLabel, pNewIns->mfConfidence, pNewIns->mbIsthing);

			//EdgeSLAM::SemanticConfidence* conf;
			//if (!mvpConfLabels[idx]->LabelConfCount.Count(pNewIns->mnLabel)) {
			//	conf = new EdgeSLAM::SemanticConfidence(pNewIns->mbIsthing);
			//}
			//else {
			//	conf = mvpConfLabels[idx]->LabelConfCount.Get(pNewIns->mnLabel);
			//}
			//float fconf = pNewIns->mfConfidence;
			//
			//conf->Add(fconf);
			//float val = conf->conf;

			//mvpConfLabels[idx]->LabelConfCount.Update(pNewIns->mnLabel, conf);

			//if (val > mvpConfLabels[idx]->maxConf) {
			//	mvpConfLabels[idx]->label = label;
			//	mvpConfLabels[idx]->maxConf = val;
			//}
			//else {
			//	//slabel 
			//}

			mapPoints[newPair.first].push_back(mvKeyDatas[idx].pt);

			/*else {
				label = pMPi->mpConfLabel->label;
			}*/

			//smvpConfLabels[idx]->LabelConfCount.U

			//auto pair2 = vPairFrameAndBox[idx];
			//if (pid != pair2.first)
			//{
			//	std::cout << "UpdateInstanceKeyPoints????????????????????????????" << std::endl;
			//}
			//auto kpidx = pair2.second;
			//if (kpidx > pOldIns->mvKeyDatas.size())
			//	std::cout << "error index" << std::endl;
			//
			//pOldIns->mvbInlierKPs.update(kpidx, false);
			//const auto kp = pOldIns->mvKeyDatas[kpidx];
			//const auto kpUn = pOldIns->mvKeyDataUns[kpidx];
			//const cv::Mat d = pOldIns->mDescriptors.row(kpidx).clone();
			////std::cout << "c" << std::endl;
			//pNewIns->AddData(kp, kpUn, d);
			////std::cout << "d" << std::endl;
		}

		//mask filling
		for (auto pair : mapPoints) {
			auto sid = pair.first;
			auto vecPoints = pair.second;

			float min_x = 1000.0;
			float max_x = 0.0;
			float min_y = 1000.0;
			float max_y = 0.0;

			for (auto pt : vecPoints) {
				if (pt.x > max_x) {
					max_x = pt.x;
				}
				if (pt.x < min_x) {
					min_x = pt.x;
				}
				if (pt.y > max_y) {
					max_y = pt.y;
				}
				if (pt.y < min_y) {
					min_y = pt.y;
				}
			}
			min_x -= 2;
			min_y -= 2;
			max_x += 2;
			max_y += 2;
			if (min_x < 0.0)
				min_x = 0.0;
			if (max_x > w)
				max_x = w;
			if (min_y < 0.0)
				min_y = 0.0;
			if (max_y > h)
				max_y = h;
			auto rect = cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
			{
				std::unique_lock<std::mutex> lock(mMutexInstance);
				cv::rectangle(seg, rect, cv::Scalar(sid, 0, 0), -1);
			}
			//seg(rect) = sid;
			//std::cout << "new ins rect = " << rect << std::endl;
		}
	}

	void BoxFrame::UpdateInstances(BoxFrame* pTarget, const std::map < std::pair<int, int>, std::pair<int, int>>& mapChanged) {
		
		auto pPrevKF = mpRefKF;
		auto pCurrKF = pTarget->mpRefKF;

		for (auto pair : mapChanged) {
			auto oldpair = pair.first;
			auto newpair = pair.second;

			SegInstance* prevIns = nullptr;
			SegInstance* currIns = nullptr;
			if (oldpair.first != newpair.first) {
				//currIns ȹ��
				//prev ����
				currIns = pTarget->mmpBBs[oldpair.second];
				prevIns = new SegInstance(this, pPrevKF->fx, pPrevKF->fy, pPrevKF->cx, pPrevKF->cy, currIns->mpConfLabel->label, currIns->mpConfLabel->maxConf, currIns->mbIsthing, this->mpDevice, false);
				prevIns->SetPose(GetPose());
				prevIns->mStrLabel = currIns->mStrLabel;
				mmpBBs[newpair.first] = prevIns;
			}
			if (oldpair.second != newpair.second) {
				/*if(oldpair.first != newpair.first){
					std::cout << "Ins::Update::error::cid" << std::endl;
				}
				continue;*/

				//prev ȹ��
				//curr ����
				prevIns = mmpBBs[oldpair.first];
				currIns = new SegInstance(pTarget, pCurrKF->fx, pCurrKF->fy, pCurrKF->cx, pCurrKF->cy, prevIns->mpConfLabel->label, prevIns->mpConfLabel->maxConf, prevIns->mbIsthing, pTarget->mpDevice, false);
				currIns->SetPose(pTarget->GetPose());
				currIns->mStrLabel = prevIns->mStrLabel;
				pTarget->mmpBBs[newpair.second] = currIns;
			}
			//std::cout << "add new ins = " <<mnId<<" " << newpair.first << std::endl;
			/*auto currIns = pTarget->mmpBBs[oldpair.second];
			auto prevIns = new ObjectSLAM::SegInstance(this, pPrevKF->fx, pPrevKF->fy, pPrevKF->cx, pPrevKF->cy, currIns->mpConfLabel->label, currIns->mpConfLabel->maxConf, currIns->mbIsthing, this->mpDevice);
			prevIns->SetPose(GetPose());
			prevIns->mStrLabel = currIns->mStrLabel;
			mmpBBs[newpair.first] = prevIns;*/

			prevIns->UpdateInstance(currIns);
			currIns->UpdateInstance(prevIns);
		}
	}

	void BoxFrame::UpdateInstances(BoxFrame* pTarget, const std::map<int, int>& mapLinkIDs) {

		auto pPrevKF = mpRefKF;
		auto pCurrKF = pTarget->mpRefKF;

		for (auto pair : mapLinkIDs) {
			auto pid = pair.first;
			auto cid = pair.second;

			SegInstance* prevIns = nullptr; 
			SegInstance* currIns = nullptr;
			if (!mmpBBs.count(pid))
			{
			}
			else {
				prevIns = mmpBBs[pid];
			}
			if (!pTarget->mmpBBs.count(cid))
			{
			}
			else {
				currIns = pTarget->mmpBBs[cid];
			}

			/*if (!prevIns)
				std::cout << "null prev ins" << std::endl;*/
			if (!currIns)
				std::cout << "null curr ins" << std::endl;

			//if (!mmpBBs.count(pid))
			//{
			//	//std::cout << "add new ins = " << pid << std::endl;
			//	prevIns = new ObjectSLAM::SegInstance(this, pPrevKF->fx, pPrevKF->fy, pPrevKF->cx, pPrevKF->cy, currIns->mnLabel, currIns->mfConfidence, currIns->mbIsthing, this->mpDevice);
			//	prevIns->SetPose(GetPose());
			//	mmpBBs[pid] = prevIns;
			//}
			//if (!pTarget->mmpBBs.count(cid))
			//{
			//	currIns = new ObjectSLAM::SegInstance(pTarget, pCurrKF->fx, pCurrKF->fy, pCurrKF->cx, pCurrKF->cy, prevIns->mnLabel, prevIns->mfConfidence, prevIns->mbIsthing, pTarget->mpDevice);
			//	currIns->SetPose(pTarget->GetPose());
			//	pTarget->mmpBBs[cid] = currIns;
			//}

			if (!prevIns)
				std::cout << "null prev ins" << std::endl;
			if (!currIns)
				std::cout << "null curr ins" << std::endl;

			//prevIns->UpdateInstance(currIns);
			//currIns->UpdateInstance(prevIns);

			/*if (prevIns->mnConnected == 1 && currIns->mnConnected == 1) {
				std::cout << "global instance test = " << std::endl;
			}*/

		}

	}

	void BoxFrame::MatchingFrameWithDenseOF(BoxFrame* pTarget, std::vector<cv::Point2f>& vecPoints1, std::vector<cv::Point2f>& vecPoints2, int scale) {

		std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

		cv::Mat gray1, gray2;
		if (scale > 1) 
		{
			cv::resize(gray, gray1, gray.size() / scale);
			cv::resize(pTarget->gray, gray2, gray.size() / scale);
		}
		cv::Mat flow;
		cv::calcOpticalFlowFarneback(gray1,gray2, flow, 0.5, 3, 15, 3, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN);

		for (int x = 0; x < flow.cols; x++) {
			for (int y = 0; y < flow.rows; y++) {
				int nx = x * scale;
				int ny = y * scale;

				if (nx >= gray.cols || ny >= gray.rows)
					continue;
				
				float fx = flow.at<cv::Vec2f>(y, x).val[0] * scale;
				float fy = flow.at<cv::Vec2f>(y, x).val[1] * scale;

				if (abs(fx) < 0.001 && abs(fy) < 0.001)
					continue;

				cv::Point2f pt1(nx, ny);
				cv::Point2f pt2(nx + fx, ny + fy);
				//cv::line(img1, pt1, pt2, cv::Scalar(255, 0, 0), 2);

				if (pt2.x < 0 || pt2.y < 0 || pt2.x >= pTarget->gray.cols || pt2.y >= pTarget->gray.rows)
					continue;

				vecPoints1.push_back(pt1);
				vecPoints2.push_back(pt2);
				
				/*int label = labeled1.at<uchar>(ny, nx) + 1;
				if (label <= 0)
				{
					continue;
				}
				cv::circle(img2, pt2, 5, SemanticColors[label], -1);*/
			}
		}

		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "dense test = " << du_test1 << std::endl;
	}
	
	void BoxFrame::MatchingWithFrame(EdgeSLAM::Frame* pTarget, const cv::Mat& fgray, std::vector<int>& vecInsIDs, std::map<int, int>& mapInsNLabel, std::vector<cv::Point2f>& vecCorners) {
		
		//std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
		
		std::vector<cv::Point2f> vecFrameCorners, vexBoxFrameCorners;
		std::vector<int> idxs;
		for (int i = 0; i < pTarget->N; i++) {
		 	vecFrameCorners.push_back(pTarget->mvKeys[i].pt);
		}
		vecInsIDs = std::vector<int>(pTarget->N, -1);
		std::vector<uchar> features_found;

		if (vecFrameCorners.size() < 10)
			return;

		int win_size = 10;
		cv::Mat pgray = fgray.clone();
		cv::Mat cgray = gray.clone();
		cv::calcOpticalFlowPyrLK(
			pgray,                         // Previous image
			cgray,                         // Next image
			vecFrameCorners,                     // Previous set of corners (from imgA)
			vecCorners,                     // Next set of corners (from imgB)
			features_found,               // Output vector, each is 1 for tracked
			cv::noArray(),                // Output vector, lists errors (optional)
			cv::Size(win_size * 2 + 1, win_size * 2 + 1),  // Search window size
			5,                            // Maximum pyramid level to construct
			cv::TermCriteria(
				cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
				20,                         // Maximum number of iterations
				0.3                         // Minimum change per iteration
			)
		);

		//�������� ������ �̿��� ��Ī ���� üũ
		cv::Mat T1 = pTarget->GetPose();
		cv::Mat R1 = T1.rowRange(0, 3).colRange(0, 3);
		cv::Mat t1 = T1.rowRange(0, 3).col(3);
		const cv::Mat T2 = this->GetPose();
		cv::Mat R2 = T2.rowRange(0, 3).colRange(0, 3);
		cv::Mat t2 = T2.rowRange(0, 3).col(3);
		const cv::Mat K1 = pTarget->K.clone();
		const cv::Mat K2 = this->K.clone();
		cv::Mat F12 = CommonUtils::Geometry::ComputeF12(R1, t1, R2, t2, K1, K2);

		int nfound = pTarget->N;

		int ntest = 0;
		//��Ī ���, ��Ī ��ġ, �ν��Ͻ� ���̵� tuple�� �����ϱ�
		
		for (int i = 0; i < nfound; ++i) {
			if (!features_found[i]) {
				continue;
			}
			auto framePt = vecFrameCorners[i];
			auto boxframePt = vecCorners[i];

			auto kp = pTarget->mvKeys[i];

			if (boxframePt.x < 20 || boxframePt.x >= cgray.cols - 20 || boxframePt.y < 20 || boxframePt.y >= cgray.rows - 20)
				continue;

			//epipolar ����
			if (!CommonUtils::Geometry::CheckDistEpipolarLine(framePt, boxframePt, F12, pTarget->mvLevelSigma2[kp.octave]))
				continue;

			auto cid = this->GetInstance(boxframePt);

			if (!this->mmpBBs.count(cid))
			{
				std::cout << "box error = " << cid << std::endl;
			}
			int label = this->mmpBBs[cid]->mpConfLabel->label;

			vecInsIDs[i] = cid;
			mapInsNLabel[cid] = label;
		}

		/*std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		auto du_test1 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		std::cout << "frame matching test = " << du_test1 << std::endl;*/
	}

	void BoxFrame::MatchingWithFrame(BoxFrame* pTarget, std::vector<std::pair<int, int>>& vecPairMatchIndex){
		std::vector<cv::Point2f> vecPrevCorners, vecCurrCorners;
		//ConvertInstanceToFrame(vecPairPointIdxInBox, vecPrevCorners);

		std::vector<int> idxs;
		for (int i = 0; i < N; i++) {
			if (mvnInsIDs[i] >= 0) {
				vecPrevCorners.push_back(mvKeyDatas[i].pt);
				idxs.push_back(i);
			}
		}

		std::vector<uchar> features_found;

		if (vecPrevCorners.size() < 10)
			return;

		int win_size = 10;
		cv::Mat pgray = gray.clone();
		cv::Mat cgray = pTarget->gray.clone();
		cv::calcOpticalFlowPyrLK(
			pgray,                         // Previous image
			cgray,                         // Next image
			vecPrevCorners,                     // Previous set of corners (from imgA)
			vecCurrCorners,                     // Next set of corners (from imgB)
			features_found,               // Output vector, each is 1 for tracked
			cv::noArray(),                // Output vector, lists errors (optional)
			cv::Size(win_size * 2 + 1, win_size * 2 + 1),  // Search window size
			5,                            // Maximum pyramid level to construct
			cv::TermCriteria(
				cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
				20,                         // Maximum number of iterations
				0.3                         // Minimum change per iteration
			)
		);

		//�������� ������ �̿��� ��Ī ���� üũ
		cv::Mat T1 = GetPose();
		cv::Mat R1 = T1.rowRange(0, 3).colRange(0, 3);
		cv::Mat t1 = T1.rowRange(0, 3).col(3);
		const cv::Mat T2 = pTarget->GetPose();
		cv::Mat R2 = T2.rowRange(0, 3).colRange(0, 3);
		cv::Mat t2 = T2.rowRange(0, 3).col(3);
		const cv::Mat K2 = pTarget->mpRefKF->K.clone();
		cv::Mat F12 = CommonUtils::Geometry::ComputeF12(R1, t1, R2, t2, K, K2);

		int nfound = vecPrevCorners.size();

		int ntest = 0;
		//��Ī ���, ��Ī ��ġ, �ν��Ͻ� ���̵� tuple�� �����ϱ�
		std::unique_lock<std::mutex> lock(mMutexInstance);
		for (int i = 0; i < nfound; ++i) {
			if (!features_found[i]) {
				continue;
			}
			auto currPt = vecCurrCorners[i];

			//��ũ���� ��� ������ ���� ���� Ű����Ʈ ����
			if (currPt.x <= 0 || currPt.x >= cgray.cols || currPt.y <= 0 || currPt.y >= cgray.rows)
				continue;
			int idx2 = pTarget->mUsed.at<int>(cv::Point(currPt));
			if (idx2 <=0) {
				continue;
			}
			idx2--;
			auto prevPt = vecPrevCorners[i];
			int idx1 = idxs[i];
			
			auto kp1 = mvKeyDatas[idx1];
			auto kp2 = pTarget->mvKeyDatas[idx2];

			//epipolar ����
			if (!CommonUtils::Geometry::CheckDistEpipolarLine(prevPt, currPt, F12, mpRefKF->mvLevelSigma2[kp1.octave]))
				continue;

			//��Ī ���
			//�ν��Ͻ� ����
			//auto pid = this->GetInstance(prevPt);
			//auto cid = pTarget->GetInstance(currPt);

			vecPairMatchIndex.push_back(std::make_pair(idx1, idx2));

			/*vecIDXs.push_back(idx);
			vecPairMatches.push_back(std::make_pair(pid, cid));
			vecPairVisualizedMatches.push_back(std::make_pair(kp.pt, pt));*/
		}
	}

	void BoxFrame::MatchingWithFrame(BoxFrame* pTarget, std::vector<int>& vecIDXs, std::vector<std::pair<int, int>>& vecPairMatches, std::vector<std::pair<cv::Point2f, cv::Point2f>>& vecPairVisualizedMatches) {
		
		std::vector<cv::Point2f> vecPrevCorners, vecCurrCorners;
		//ConvertInstanceToFrame(vecPairPointIdxInBox, vecPrevCorners);

		std::vector<int> idxs;
		for (int i = 0; i < N; i++) {
			if (mvnInsIDs[i] >= 0){
				vecPrevCorners.push_back(mvKeyDatas[i].pt);
				idxs.push_back(i);
			}
		}

		std::vector<uchar> features_found;
		
		if (vecPrevCorners.size() < 10)
			return;

		int win_size = 10;
		cv::Mat pgray = gray.clone();
		cv::Mat cgray = pTarget->gray.clone();
		cv::calcOpticalFlowPyrLK(
			pgray,                         // Previous image
			cgray,                         // Next image
			vecPrevCorners,                     // Previous set of corners (from imgA)
			vecCurrCorners,                     // Next set of corners (from imgB)
			features_found,               // Output vector, each is 1 for tracked
			cv::noArray(),                // Output vector, lists errors (optional)
			cv::Size(win_size * 2 + 1, win_size * 2 + 1),  // Search window size
			5,                            // Maximum pyramid level to construct
			cv::TermCriteria(
				cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
				20,                         // Maximum number of iterations
				0.3                         // Minimum change per iteration
			)
		);

		//�������� ������ �̿��� ��Ī ���� üũ
		cv::Mat T1 = GetPose();
		cv::Mat R1 = T1.rowRange(0, 3).colRange(0, 3);
		cv::Mat t1 = T1.rowRange(0, 3).col(3);
		const cv::Mat T2 = pTarget->GetPose();
		cv::Mat R2 = T2.rowRange(0, 3).colRange(0, 3);
		cv::Mat t2 = T2.rowRange(0, 3).col(3);
		const cv::Mat K2 = pTarget->mpRefKF->K.clone();
		cv::Mat F12 = CommonUtils::Geometry::ComputeF12(R1, t1, R2, t2, K, K2);

		int nfound = vecPrevCorners.size();
		
		int ntest = 0;
		//��Ī ���, ��Ī ��ġ, �ν��Ͻ� ���̵� tuple�� �����ϱ�
		std::unique_lock<std::mutex> lock(mMutexInstance);
		for (int i = 0; i < nfound; ++i) {
			if (!features_found[i]) {
				continue;
			} 
			auto pt = vecCurrCorners[i];

			//��ũ���� ��� ������ ���� ���� Ű����Ʈ ����
			if (pt.x < 20 || pt.x >= cgray.cols - 20 || pt.y < 20 || pt.y >= cgray.rows - 20)
				continue;
			if (pTarget->mUsed.at<int>(cv::Point(pt)) > 0) {
				ntest++;
			}
			auto prevPt = vecPrevCorners[i];

			int idx = idxs[i];
			auto kp = mvKeyDatas[idx];
			
			//epipolar ����
			if (!CommonUtils::Geometry::CheckDistEpipolarLine(prevPt, pt, F12, mpRefKF->mvLevelSigma2[kp.octave]))
				continue;

			//��Ī ���
			//�ν��Ͻ� ����
			auto pid = mvnInsIDs[idx];
			auto cid = pTarget->GetInstance(pt);

			vecIDXs.push_back(idx);
			vecPairMatches.push_back(std::make_pair(pid, cid));
			vecPairVisualizedMatches.push_back(std::make_pair(kp.pt, pt));
		}
		//std::cout << vecIDXs.size() << " " << ntest << std::endl;
	}

	//�̹������� ��Ƽ���÷ο�� ��Ī�� ����Ʈ�� ��ġ �����ϴ� �ν��Ͻ� ���̵� �˷���.
	void BoxFrame::MatchingWithFrame(const cv::Mat& currGray, const cv::Mat& T2, const cv::Mat& K2, std::vector<int>& vecIDXs, std::vector<std::pair<int, cv::Point2f>>& vecPairMatches) {
		
		std::vector<std::pair<int, int>> vecPairPointIdxInBox;
		std::vector<cv::Point2f> vecPrevCorners, vecCurrCorners;
		ConvertInstanceToFrame(vecPairPointIdxInBox, vecPrevCorners);
		
		std::vector<uchar> features_found;
		
		if (vecPrevCorners.size() < 10)
			return;

		int win_size = 10;
		cv::Mat pgray = gray.clone();
		cv::Mat cgray = currGray.clone();
		cv::calcOpticalFlowPyrLK(
			pgray,                         // Previous image
			cgray,                         // Next image
			vecPrevCorners,                     // Previous set of corners (from imgA)
			vecCurrCorners,                     // Next set of corners (from imgB)
			features_found,               // Output vector, each is 1 for tracked
			cv::noArray(),                // Output vector, lists errors (optional)
			cv::Size(win_size * 2 + 1, win_size * 2 + 1),  // Search window size
			5,                            // Maximum pyramid level to construct
			cv::TermCriteria(
				cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
				20,                         // Maximum number of iterations
				0.3                         // Minimum change per iteration
			)
		);
		
		//�������� ������ �̿��� ��Ī ���� üũ
		cv::Mat T1 = GetPose();
		cv::Mat R1 = T1.rowRange(0, 3).colRange(0, 3);
		cv::Mat t1 = T1.rowRange(0, 3).col(3);
		cv::Mat R2 = T2.rowRange(0, 3).colRange(0, 3);
		cv::Mat t2 = T2.rowRange(0, 3).col(3);
		cv::Mat F12 = CommonUtils::Geometry::ComputeF12(R1, t1, R2, t2, K, K2);
		
		int nfound = vecPrevCorners.size();
		//��Ī ���, ��Ī ��ġ, �ν��Ͻ� ���̵� tuple�� �����ϱ�
		
		for (int i = 0; i < nfound; ++i) {
			if (!features_found[i]) {
				continue;
			}
			auto pt = vecCurrCorners[i];
			//��ũ���� ��� ������ ���� ���� Ű����Ʈ ����
			if (pt.x < 20 || pt.x >= currGray.cols - 20 || pt.y < 20 || pt.y >= currGray.rows - 20)
				continue;

			auto prevPt = vecPrevCorners[i];
			
			auto prevPair = vecPairPointIdxInBox[i];
			int prevId = prevPair.first;
			int prevIdx = prevPair.second;

			//int tempID = seg.at<uchar>(prevPt);
			
			auto kp = mmpBBs[prevId]->mvKeyDatas[prevIdx];
			//auto op = mmpBBs[prevId]->mvpMapDatas.get(prevIdx);

			//epipolar ����
			if (!CommonUtils::Geometry::CheckDistEpipolarLine(prevPt, pt, F12, mpRefKF->mvLevelSigma2[kp.octave]))
				continue;

			////���� �����ӿ��� Ű����Ʈ ����
			//tempMatchingPrevKP.push_back(kp);
			//tempPrevDesc.push_back(mmpBBs[prevId]->mDescriptors.row(prevIdx));

			//kp.pt = pt;
			//tempMatchingCurrKP.push_back(kp);

			//��Ī ���
			vecIDXs.push_back(i);

			//�ν��Ͻ� ����
			auto pid = this->GetInstance(prevPt);
			//auto cid = pNewBF->seg.at<uchar>(pt);
			vecPairMatches.push_back(std::make_pair(prevId, pt));

		}
		
	}

	void BoxFrame::InitLabelCount(int N) {
		matLabelCount = cv::Mat::zeros(N, 1, CV_16UC1);
		for (auto pair : mmpBBs) {
			int id = pair.first;
			auto pIns = pair.second;
			int label = pIns->mpConfLabel->label;
			if (pIns->mbIsthing)
				label += 100;
			matLabelCount.at<ushort>(label) = matLabelCount.at<ushort>(label)+1;
		}
	}

	void BoxFrame::ConvertInstanceToFrame(std::vector<std::pair<int, int>>& vPairFrameAndBox, std::vector<cv::Point2f>& vecCorners) {
		
		for (auto pair : mmpBBs) {
			int sid = pair.first;
			auto pIns = pair.second;
			
			/*if (sid < 0 || sid > 200)
			{
				std::cout << "?????????????" << std::endl << std::endl << std::endl << std::endl;
				continue;
			}*/
			
			/*if (!pIns)
				std::cout << "null instance" << std::endl;
			std::cout << "instance test = " << pIns->N <<" "<<pIns->mvbInlierKPs.size() << std::endl;*/

			for (int j = 0; j < pIns->N; j++) {

				if (!pIns->mvbInlierKPs.get(j))
					continue;

				auto kp = pIns->mvKeyDatas[j];
				vPairFrameAndBox.push_back(std::make_pair(sid, j));
				vecCorners.push_back(kp.pt);
				//mDescriptors.push_back(pIns->mDescriptors.row(j));
			}
		}

	}

	int BoxFrame::GetFrameInstanceId(EdgeSLAM::MapPoint* pMP) {
		if (!mpRefKF)
			return -1;
		auto pKF = mpRefKF;
		int idx = pMP->GetIndexInKeyFrame(pKF);
		if (idx < 0)
			return -1;
		auto pt = pKF->mvKeys[idx].pt;
		auto sid = GetInstance(pt);
		if (!mmpBBs.count(sid))
			return -1;
		return sid;
	}
	
	SegInstance* BoxFrame::GetFrameInstance(EdgeSLAM::MapPoint* pMP) {
		if (!mpRefKF)
			return nullptr;
		auto pKF = mpRefKF;
		int idx = pMP->GetIndexInKeyFrame(pKF);
		if (idx < 0)
			return nullptr;
		auto pt = pKF->mvKeys[idx].pt;
		auto sid = GetInstance(pt);
		if (!mmpBBs.count(sid))
			return nullptr;
		return mmpBBs[sid];
	}

	void BoxFrame::Copy(EdgeSLAM::Frame* pF) {
		
		for (int i = 0; i < pF->N; i++) {
			auto kp = pF->mvKeys[i];
			mvKeyDatas.push_back(kp);
			//�ش� ��Ʈ������ kp�� �ε����� ����.
			//0���� �����̶� +1�� ��. �� ��Ʈ���� �̿�� -1 �ؾ� ��.
			mUsed.at<int>(cv::Point(kp.pt)) = i + 1;
		}
		
		mDescriptors = pF->mDescriptors.clone();
		Init();
		
	}

	void BoxFrame::ConvertBoxToFrame(int w, int h) {
		/*cv::Mat used = cv::Mat::zeros(h,w, CV_8UC1);
		cv::Mat tempPrevDesc = cv::Mat::zeros(0, 32, CV_8UC1);
		mvPairFrameAndBox.clear();
		for (int i = 0; i < mvpBBs.size(); i++) {
			auto pPrevBB = mvpBBs[i];
			for (int j = 0; j < pPrevBB->N; j++) {
				auto kp = pPrevBB->mvKeyDatas[j];
				auto pt = pPrevBB->mvKeyDatas[j].pt;
				if (used.at<uchar>(pt))
					continue;
				used.at<uchar>(pt)++;
				mvPairFrameAndBox.push_back(std::make_pair(i, j));
				mvKeyDatas.push_back(kp);
				mDescriptors.push_back(pPrevBB->mDescriptors.row(j));
			}
		}*/
	}
	void BoxFrame::Init() {
		
		mpKC->Init(mpCamera->bDistorted, mpCamera->K, mpCamera->D);
		
		mpSC->Init(N);
		/*UndistortKeyDatas(mpCamera->K, mpCamera->D);*/
		
		bool bDepth = mpCamera->mCamSensor == BaseSLAM::CameraSensor::RGBD;
		if (bDepth) {
			mpSC->ComputeStereoFromRGBD(depth, mbf, mpKC);
		}
		
		mvnInsIDs = std::vector<int>(N, -1);
		mvpConfLabels = std::vector<EdgeSLAM::SemanticConfLabel*>(N, nullptr);
		/*for (int i = 0; i < N; i++){
			mvpConfLabels[i] = new EdgeSLAM::SemanticConfLabel();
		}*/

		////box �ʱ�ȭ
		//for (int j = 0; j < N; j++) {

		//	auto pt = mvKeyDatas[j].pt;
		  
		//	for (int i = 0; i < mvpBBs.size(); i++) {
		//		auto pBBox = mvpBBs[i];
		//		if (!pBBox->mRect.contains(pt))
		//			continue;
		//		cv::Mat row = mDescriptors.row(j);
		//		pBBox->mvKPs.push_back(mvKeyDatas[j]);
		//		pBBox->mvKPsUn.push_back(mvKeyDataUns[j]);
		//		pBBox->desc.push_back(row.clone());
		//		/*size_t idx = pBBox->mvKPs.size();
		//		pBBox->mvIDXs.push_back(j);
		//		pBBox->mapIDXs[j] = idx;*/
		//		if(bDepth)
		//			pBBox->mvDepth.push_back(mvDepth[j]);
		//	}
		//}

		//cv::Mat T= mpPose->GetPose();

		//for (int i = 0; i < mvpBBs.size(); i++) {
		//	auto pBBox = mvpBBs[i];
		//	pBBox->mpWorldPose->SetPose(T);
		//}

	}

	void BoxFrame::BaseObjectRegistration(EdgeSLAM::KeyFrame* pNewKF) {
		//
		auto vpBBs = mvpBBs;
		for (int i = 0; i < pNewKF->N; i++) {
			auto kp = pNewKF->mvKeys[i];
			auto mp = pNewKF->mvpMapPoints.get(i);
			bool bmp = false;
			if (!mp || mp->isBad()) {

			}
			else {
				bmp = true;
			}
			for (int j = 0; j < vpBBs.size(); j++) {
				auto pBB = vpBBs[j];
				if (pBB->mRect.contains(kp.pt)) {
					pBB->n1++;
					cv::circle(img, kp.pt, 5, cv::Scalar(0, 255, 0), -1);
					if (bmp)
					{
						pBB->n2++;
						pBB->origin += mp->GetWorldPos();
						cv::circle(img, kp.pt, 5, cv::Scalar(0, 0, 255), -1);
					}
				}
			}
		}
		//
		std::cout << "Obj::Registration::Baseline test = " << pNewKF->mnId << std::endl;
		for (int j = 0; j < vpBBs.size(); j++) {
			auto pBB = vpBBs[j];
			
			cv::rectangle(img, pBB->mRect, cv::Scalar(0, 255, 0), 2);
			pBB->origin /= pBB->n2;

			std::cout << j << " " <<" "<<pBB->origin.t() <<" == " << pBB->n1 << " " << pBB->n2 << std::endl;
		}
	}

	void BoxFrame::InitInstance(const cv::Mat& mapInstance){
		{
			std::unique_lock<std::mutex> lock(mMutexInstance);
			seg = mapInstance.clone();
		}
	}

	int BoxFrame::GetInstance(const cv::Point& pt){
		std::unique_lock<std::mutex> lock(mMutexInstance);
		return seg.at<uchar>(pt);
	}
	void BoxFrame::SetInstance(const cv::Point& pt, int _sid){
		std::unique_lock<std::mutex> lock(mMutexInstance);
		seg.at<uchar>(pt) = _sid;
	}

	//frame�� map�� ������ �ʿ���.
	//frame = prev ins + curr ins + global
	//map   = global   + curr ins ==> global�� �̸� curr�� map instance�� �߰�?
	void InstanceMask::UpdateAssociation(InstanceMask* pPrev, BoxFrame* pCurrBF, BoxFrame* pPrevBF, ObjectSLAM* ObjSystem) {

		auto vecResAsso = this->mvResAsso.get();

		//for (auto assores : vecResAsso)
		//{
		//	if (!assores->res)
		//		continue;
		//	
		//	int id1 = -1;

		//	if (assores->nDataType == 1)
		//	{
		//		id1 = assores->id1;
		//	}
		//	int id2 = assores->id2;

		//	if (id1 == 0 || id2 == 0)
		//	{
		//		std::cout << "object matching error????" << std::endl;
		//		continue;
		//	}

		//	///*if (pPrevG && pCurrG && pPrevG->mnId != pCurrG->mnId)
		//	//{
		//	//	vecNeedMerge.push_back(std::make_pair(pPrevG, pCurrG));
		//	//}*/

		//	FrameInstance* pPrevIns = nullptr;
		//	GlobalInstance* pPrevG = nullptr;
		//	if(assores->nDataType == 1){
		//		pPrevIns = pPrev->FrameInstances.Get(id1);
		//		pPrevG = pPrev->MapInstances.Get(id1);
		//	}
		//	auto pCurrIns = this->FrameInstances.Get(id2);
		//	auto pCurrG = this->MapInstances.Get(id2);

		//	//�۷ι� �ν��Ͻ� ���� 
		//	if (assores->nDataType == 1)
		//	{
		//		if (pPrevG && !pCurrG)
		//		{
		//			this->MapInstances.Update(id2, pPrevG);
		//			pPrevG->Connect(pCurrIns, pCurrBF, id2);
		//			pCurrG = pPrevG;
		//			//pPrevG->EIFFilterOutlier();
		//			//pPrevG->Connect(pNewBF, id2);
		//			//pPrevG->AddMapPoints(pCurrIns->setMPs);
		//		}
		//		if (!pPrevG && pCurrG)
		//		{
		//			pPrev->MapInstances.Update(id1, pCurrG);
		//			pCurrG->Connect(pPrevIns, pPrevBF, id1);
		//			//pCurrG->EIFFilterOutlier();
		//			//pCurrG->Connect(pPrevBF, id1);
		//			//pCurrG->AddMapPoints(pPrevIns->setMPs);
		//		}

		//		//�۷ι� �ν��Ͻ� ����
		//		if (!pPrevG && !pCurrG)
		//		{
		//			auto pGlobal = new GlobalInstance();
		//			pPrev->MapInstances.Update(id1, pGlobal);
		//			this->MapInstances.Update(id2, pGlobal);

		//			pGlobal->Connect(pPrevIns, pPrevBF, id1);
		//			pGlobal->Connect(pCurrIns, pCurrBF, id2);
		//			pPrevG = pGlobal;
		//			pCurrG = pGlobal;
		//			//pGlobal->EIFFilterOutlier();
		//			//pGlobal->Connect(pPrevBF, id1);
		//			//pGlobal->Connect(pNewBF, id2);
		//			//pGlobal->AddMapPoints(pCurrIns->setMPs);
		//			//pGlobal->AddMapPoints(pPrevIns->setMPs);
		//		}

		//		pCurrG->Update(pCurrBF->mpRefKF);

		//		if (pPrevG)
		//		{
		//			ObjSystem->vecObjectAssoRes.push_back(assores->print(id1, pPrevG->mnId));
		//		}
		//		else
		//			ObjSystem->vecObjectAssoRes.push_back(assores->print(id2));
		//	}
		//	else
		//	{
		//		//�� �ν��Ͻ��� �ٷ� �߰��� ��
		//		if(pCurrG){
		//			ObjSystem->vecObjectAssoRes.push_back(assores->print(id1, pCurrG->mnId));
		//		}
		//	}
		//	
		//	
		//}

	}

}