#include <BoxFrame.h>
#include <BoundingBox.h>
#include <SegInstance.h>
#include <KeyFrame.h>
#include <Frame.h>
#include <MapPoint.h>
#include <Utils_Geometry.h>

namespace ObjectSLAM {
	BoxFrame::BoxFrame(int _id) :BaseSLAM::AbstractFrame(_id)
	{}
	BoxFrame::BoxFrame(int _id, const int w, const int h, BaseSLAM::BaseDevice* Device, BaseSLAM::AbstractPose* _Pose) : BaseSLAM::AbstractFrame(Device, _Pose, _id), BaseSLAM::KeyPointContainer(mpCamera), BaseSLAM::StereoDataContainer(), //mUsed(cv::Mat::zeros(h, w, CV_8UC1)),
		mpKC(this), mpSC(this), mpRefKF(nullptr), mpDevice(Device)
	{}
	BoxFrame::~BoxFrame() {
		std::vector<BoundingBox*>().swap(mvpBBs);
		img.release();
		labeled.release();
		depth.release();
	}

	void BoxFrame::UpdateInstanceKeyPoints(const std::vector<std::pair<int, int>>& vecMatches, const std::vector<int>& vecIDXs, const std::vector<std::pair<int, int>>& vPairFrameAndBox, std::map<std::pair<int,int>, std::pair<int, int>>& mapChangedIns) {
		
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
			
			mvLabels[idx] = newPair.first;

			auto pair2 = vPairFrameAndBox[idx];
			if (pid != pair2.first)
			{
				std::cout << "UpdateInstanceKeyPoints????????????????????????????" << std::endl;
			}
			auto kpidx = pair2.second;
			if (kpidx > pOldIns->mvKeyDatas.size())
				std::cout << "error index" << std::endl;
			
			pOldIns->mvbInlierKPs.update(kpidx, false);
			const auto kp = pOldIns->mvKeyDatas[kpidx];
			const auto kpUn = pOldIns->mvKeyDataUns[kpidx];
			const cv::Mat d = pOldIns->mDescriptors.row(kpidx).clone();
			//std::cout << "c" << std::endl;
			pNewIns->AddData(kp, kpUn, d);
			//std::cout << "d" << std::endl;
		}
	}

	void BoxFrame::UpdateInstances(BoxFrame* pTarget, const std::map < std::pair<int, int>, std::pair<int, int>>& mapChanged) {
		
		auto pPrevKF = mpRefKF;
		auto pCurrKF = pTarget->mpRefKF;

		for (auto pair : mapChanged) {
			auto oldpair = pair.first;
			auto newpair = pair.second;

			if (oldpair.first != newpair.first) {

			}
			if (oldpair.second != newpair.second) {
				std::cout << "Ins::Update::error::cid" << std::endl;
			}
			//std::cout << "add new ins = " <<mnId<<" " << newpair.first << std::endl;
			auto currIns = pTarget->mmpBBs[oldpair.second];
			auto prevIns = new ObjectSLAM::SegInstance(this, pPrevKF->fx, pPrevKF->fy, pPrevKF->cx, pPrevKF->cy, currIns->mnLabel, currIns->mfConfidence, currIns->mbIsthing, this->mpDevice);
			prevIns->SetPose(GetPose());
			mmpBBs[newpair.first] = prevIns;
		}
	}

	void BoxFrame::UpdateInstances(BoxFrame* pTarget, const std::map<int, int>& mapLinkIDs) {

		auto pPrevKF = mpRefKF;
		auto pCurrKF = pTarget->mpRefKF;

		for (auto pair : mapLinkIDs) {
			auto pid = pair.first;
			auto cid = pair.second;

			ObjectSLAM::SegInstance* prevIns = nullptr; 
			ObjectSLAM::SegInstance* currIns = nullptr;
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

			if (!mmpBBs.count(pid))
			{
				//std::cout << "add new ins = " << pid << std::endl;
				prevIns = new ObjectSLAM::SegInstance(this, pPrevKF->fx, pPrevKF->fy, pPrevKF->cx, pPrevKF->cy, currIns->mnLabel, currIns->mfConfidence, currIns->mbIsthing, this->mpDevice);
				prevIns->SetPose(GetPose());
				mmpBBs[pid] = prevIns;
			}
			if (!pTarget->mmpBBs.count(cid))
			{
				currIns = new ObjectSLAM::SegInstance(pTarget, pCurrKF->fx, pCurrKF->fy, pCurrKF->cx, pCurrKF->cy, prevIns->mnLabel, prevIns->mfConfidence, prevIns->mbIsthing, pTarget->mpDevice);
				currIns->SetPose(pTarget->GetPose());
				pTarget->mmpBBs[cid] = currIns;
			}

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

	void BoxFrame::MatchingWithFrame(BoxFrame* pTarget, std::vector<int>& vecIDXs, std::vector<std::pair<int, int>>& vecPairMatches, std::vector<std::pair<int, int>>& vecPairPointIdxInBox) {
		
		//std::vector<cv::Point2f> vecPrevCorners, vecCurrCorners;
		//ConvertInstanceToFrame(vecPairPointIdxInBox, vecPrevCorners);

		for (int i = 0; i < N; i++) {
			if (mvLabels[i] >= 0)
				vecPrevCorners.push_back(mvKeyDatas[i].pt);
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

		//에피폴라 제약을 이용한 매칭 에러 체크
		cv::Mat T1 = GetPose();
		cv::Mat R1 = T1.rowRange(0, 3).colRange(0, 3);
		cv::Mat t1 = T1.rowRange(0, 3).col(3);
		const cv::Mat T2 = pTarget->GetPose();
		cv::Mat R2 = T2.rowRange(0, 3).colRange(0, 3);
		cv::Mat t2 = T2.rowRange(0, 3).col(3);
		const cv::Mat K2 = pTarget->mpRefKF->K.clone();
		cv::Mat F12 = CommonUtils::Geometry::ComputeF12(R1, t1, R2, t2, K, K2);

		int nfound = vecPrevCorners.size();
		
		//매칭 결과, 매칭 위치, 인스턴스 아이디를 tuple로 저장하기
		for (int i = 0; i < nfound; ++i) {
			if (!features_found[i]) {
				continue;
			}
			auto pt = vecCurrCorners[i];
			//디스크립터 계산 가능한 영역 안의 키포인트 검출
			if (pt.x < 20 || pt.x >= cgray.cols - 20 || pt.y < 20 || pt.y >= cgray.rows - 20)
				continue;

			auto prevPt = vecPrevCorners[i];

			auto prevPair = vecPairPointIdxInBox[i];
			int prevId = prevPair.first;
			int prevIdx = prevPair.second;

			//int tempID = seg.at<uchar>(prevPt);

			auto kp = mmpBBs[prevId]->mvKeyDatas[prevIdx];
			//auto op = mmpBBs[prevId]->mvpMapDatas.get(prevIdx);

			//epipolar 제약
			if (!CommonUtils::Geometry::CheckDistEpipolarLine(prevPt, pt, F12, mpRefKF->mvLevelSigma2[kp.octave]))
				continue;

			////이전 프레임에서 키포인트 정보
			//tempMatchingPrevKP.push_back(kp);
			//tempPrevDesc.push_back(mmpBBs[prevId]->mDescriptors.row(prevIdx));

			//kp.pt = pt;
			//tempMatchingCurrKP.push_back(kp);

			//매칭 결과
			vecIDXs.push_back(i);

			//인스턴스 연결
			auto pid = seg.at<uchar>(prevPt);
			auto cid = pTarget->seg.at<uchar>(pt);
			//auto cid = pNewBF->seg.at<uchar>(pt);
			vecPairMatches.push_back(std::make_pair(pid, cid));

		}
		
	}

	//이미지에서 옵티컬플로우로 매칭된 포인트의 위치 대응하는 인스턴스 아이디를 알려줌.
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
		
		//에피폴라 제약을 이용한 매칭 에러 체크
		cv::Mat T1 = GetPose();
		cv::Mat R1 = T1.rowRange(0, 3).colRange(0, 3);
		cv::Mat t1 = T1.rowRange(0, 3).col(3);
		cv::Mat R2 = T2.rowRange(0, 3).colRange(0, 3);
		cv::Mat t2 = T2.rowRange(0, 3).col(3);
		cv::Mat F12 = CommonUtils::Geometry::ComputeF12(R1, t1, R2, t2, K, K2);
		
		int nfound = vecPrevCorners.size();
		//매칭 결과, 매칭 위치, 인스턴스 아이디를 tuple로 저장하기
		for (int i = 0; i < nfound; ++i) {
			if (!features_found[i]) {
				continue;
			}
			auto pt = vecCurrCorners[i];
			//디스크립터 계산 가능한 영역 안의 키포인트 검출
			if (pt.x < 20 || pt.x >= currGray.cols - 20 || pt.y < 20 || pt.y >= currGray.rows - 20)
				continue;

			auto prevPt = vecPrevCorners[i];
			
			auto prevPair = vecPairPointIdxInBox[i];
			int prevId = prevPair.first;
			int prevIdx = prevPair.second;

			//int tempID = seg.at<uchar>(prevPt);
			
			auto kp = mmpBBs[prevId]->mvKeyDatas[prevIdx];
			//auto op = mmpBBs[prevId]->mvpMapDatas.get(prevIdx);

			//epipolar 제약
			if (!CommonUtils::Geometry::CheckDistEpipolarLine(prevPt, pt, F12, mpRefKF->mvLevelSigma2[kp.octave]))
				continue;

			////이전 프레임에서 키포인트 정보
			//tempMatchingPrevKP.push_back(kp);
			//tempPrevDesc.push_back(mmpBBs[prevId]->mDescriptors.row(prevIdx));

			//kp.pt = pt;
			//tempMatchingCurrKP.push_back(kp);

			//매칭 결과
			vecIDXs.push_back(i);

			//인스턴스 연결
			auto pid = seg.at<uchar>(prevPt);
			//auto cid = pNewBF->seg.at<uchar>(pt);
			vecPairMatches.push_back(std::make_pair(prevId, pt));

		}
		
	}

	void BoxFrame::InitLabelCount(int N) {
		matLabelCount = cv::Mat::zeros(N, 1, CV_16UC1);
		for (auto pair : mmpBBs) {
			int id = pair.first;
			auto pIns = pair.second;
			int label = pIns->mnLabel;
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
		auto sid = seg.at<uchar>(pt);
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
		auto sid = seg.at<uchar>(pt);
		if (!mmpBBs.count(sid))
			return nullptr;
		return mmpBBs[sid];
	}

	void BoxFrame::Copy(EdgeSLAM::Frame* pF) {
		for (int i = 0; i < pF->N; i++) {
			auto kp = pF->mvKeys[i];
			pF->mvKeys.push_back(kp);
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

		mvLabels = std::vector<int>(N, -1);

		////box 초기화
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
}