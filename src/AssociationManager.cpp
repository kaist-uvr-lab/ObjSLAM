#include <AssociationManager.h>

//EdgeSLAM
#include <Utils.h>
#include <Utils_Geometry.h>
#include <SLAM.h>
#include <Camera.h>
#include <KeyFrame.h>
#include <MapPoint.h>

//ObjectSLAM
#include <FrameInstance.h>
#include <GlobalInstance.h>
#include <BoxFrame.h>
#include <ObjectSLAM.h>
#include <ObjectMatcher.h>
#include <InstanceLinker.h>
#include <AssoFramePairData.h> 
#include <ObjectRegionFeatures.h>

//가우시안 객체 맵
#include <Gaussian/GaussianObject.h>
#include <Gaussian/GaussianMapManager.h>
#include <Gaussian/Visualizer.h>
#include <Gaussian/Optimization/ObjectOptimizer.h>

namespace ObjectSLAM {
	
	ConcurrentMap<int, int> AssociationManager::DebugAssoSeg, AssociationManager::DebugAssoSAM;

	float sign(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3) {
		return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
	}
	bool pointInTriangle(const cv::Point2f& pt, const cv::Point2f& v1,
		const cv::Point2f& v2, const cv::Point2f& v3) {
		// 외적을 이용한 방법
		float d1 = sign(pt, v1, v2);
		float d2 = sign(pt, v2, v3);
		float d3 = sign(pt, v3, v1);

		bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
		bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

		// 부호가 모두 같으면 삼각형 내부에 있음
		return !(has_neg && has_pos);
	}
	bool isPointInRotatedRect(const cv::Point2f& point, const cv::RotatedRect& rect) {
		// 회전된 사각형의 꼭지점 4개 얻기
		cv::Point2f vertices[4];
		rect.points(vertices);

		// 점이 4개의 삼각형 안에 있는지 확인
		// 사각형을 두 개의 삼각형으로 나누어 확인
		return (pointInTriangle(point, vertices[0], vertices[1], vertices[2]) ||
			pointInTriangle(point, vertices[0], vertices[2], vertices[3]));
	}

	void AssociationManager::AssociationWithSeg(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM,
		const std::string& key, const std::string& mapName, const std::string& userName, AssoFramePairData* pPairData)
	{

		auto pPrevBF = pPairData->mpFrom;
		auto pNewBF = pPairData->mpTo;

		auto pCurrKF = pNewBF->mpRefKF;
		auto pPrevKF = pPrevBF->mpRefKF;

		auto pCurrSegMask = pNewBF->mapMasks.Get("yoloseg");
		auto pPrevSegMask = pPrevBF->mapMasks.Get("yoloseg");

		auto pCurrSegInstance = pCurrSegMask->FrameInstances.Get();
		auto pPrevSegInstance = pPrevSegMask->FrameInstances.Get();

		//가우시안 맵. pCurrGO는 MAP을 이용한 연결에서 연결된 경우 프레임끼리 매칭 될 때 이전 프레임에 맵을 전달하는 용도
		auto pCurrGOs = pCurrSegMask->GaussianMaps.Get();
		auto pPrevGOs = pPrevSegMask->GaussianMaps.Get();

		auto pRaftMask = pPairData->mpRaftIns;

		std::map<int, FrameInstance*> mapRaftInstance;
		const cv::Mat flow = pRaftMask->mask;

		int didx = 0;
		DebugAssoSeg.Update(didx, DebugAssoSeg.Get(didx) + 1);
		

		ConvertMaskWithRAFT(pPrevSegInstance, mapRaftInstance, pPrevKF, flow);
		for (auto pair : mapRaftInstance)
		{
			auto pid = pair.first;
			auto pRaftIns = pair.second;
			pRaftMask->FrameInstances.Update(pid, pRaftIns);
			pRaftMask->GaussianMaps.Update(pid, nullptr);
		}
		//prev의 GO, RAFT check

		didx = 1;
		DebugAssoSeg.Update(didx, DebugAssoSeg.Get(didx) + 1);
		

		//iou matching
		std::map<std::pair<int, int>, AssoMatchRes*> res, mapSuccess;
		CalculateIOU(mapRaftInstance, pCurrSegInstance, res);
		EvaluateMatchResults(res, mapSuccess);

		didx = 2;
		DebugAssoSeg.Update(didx, DebugAssoSeg.Get(didx) + 1);
		
		std::set<int> sAlready;
		for (auto pair : mapSuccess)
		{
			auto pid = pair.first.first;
			auto cid = pair.first.second;

			if (!sAlready.count(pid))
				sAlready.insert(pid);

			auto pPrevIns = pPrevSegInstance[pid];
			auto pRaftIns = mapRaftInstance[pid];
			auto pPrevGO = pPrevGOs[pid];

			auto ares = pair.second;

			pRaftMask->mapResAssociation[pid] = ares;

			if (ares->iou > 0.5)
			{
				if (cid == 0)
				{
					ares->req = true;
					ares->res = false;
					pPairData->mapReqRaft[pid] = 0;
					/*pSAMRaftMask->FrameInstances.Update(pid, pRaftIns);
					pSAMRaftMask->GaussianMaps.Update(pid, pPrevGO);
					pSAMRaftMask->mapResAssociation[pid] = ares;*/

					/*pSAMSegMask->FrameInstances.Update(pid, pPrevIns);
					pSAMSegMask->GaussianMaps.Update(pid, pPrevGO);
					pSAMSegMask->mapResAssociation[pid] = ares;*/
				}
				else {
					ares->res = true;
					ares->req = false;

					pPairData->mapRaftSeg[pid] = cid;
					/*pNewMask->FrameInstances.Update(pid, pPrevIns);
					pNewMask->GaussianMaps.Update(pid, pPrevGO);
					pNewMask->mapResAssociation[pid] = ares;*/
				}
				/*mapMapMatchRes[pPrevIns].insert(id);
				mapCurrMatchRes[cid].insert(pPrevIns);*/
			}
		}

		didx = 3;
		DebugAssoSeg.Update(didx, DebugAssoSeg.Get(didx) + 1);

		//request SAM
		cv::Mat ptdata(0, 1, CV_32FC1);
		for (auto pair : pPairData->mapReqRaft)
		{
			auto pid = pair.first;
			auto p = mapRaftInstance[pid];
			auto rect = p->rect;
			cv::Mat temp = cv::Mat::zeros(4, 1, CV_32FC1);
			temp.at<float>(0) = rect.x;
			temp.at<float>(1) = rect.y;
			temp.at<float>(2) = rect.x + rect.width;
			temp.at<float>(3) = rect.y + rect.height;
			ptdata.push_back(temp);
		}
		if (ptdata.rows > 0) {
			//reqest
			int nobj = ptdata.rows;
			ptdata.push_back(cv::Mat::zeros(1500 - nobj, 1, CV_32FC1));
			//id2 : prev frame, type : (1) frame, (2) map
			std::string tsrc = userName + ".Image." + std::to_string(nobj) + "." + std::to_string(pPairData->fromid) + "." + std::to_string((int)InstanceType::SEG);
			auto sam2key = "reqsam2";
			std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();
			auto du_upload = Utils::SendData(sam2key, tsrc, ptdata, pPairData->toid, 15, t_start.time_since_epoch().count());
		}

		for (auto pair : pPrevSegInstance)
		{
			auto pid = pair.first;
			if (!pPairData->mapReqRaft.count(pid) && !pPairData->mapRaftSeg.count(pid))
				pPairData->setSegFromFailed.insert(pid);
		}

		std::set<int> tempIDs;
		for (auto pair : pPairData->mapRaftSeg)
		{
			auto cid = pair.second;
			tempIDs.insert(cid);
		}
		for (auto pair : pCurrSegInstance)
		{
			auto cid = pair.first;
			if (!tempIDs.count(cid))
				pPairData->setSegToFailed.insert(cid);
		}

		didx = 4;
		DebugAssoSeg.Update(didx, DebugAssoSeg.Get(didx) + 1);
		

		//check sam request
		AssociationWithUncertainty(SLAM, pPairData);

		didx = 5;
		DebugAssoSeg.Update(didx, DebugAssoSeg.Get(didx) + 1);

		std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();
		AssociateLocalMapWithUncertainty(SLAM, pPairData);
		std::chrono::high_resolution_clock::time_point t_end = std::chrono::high_resolution_clock::now();
		auto du_seg = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
		//std::cout << "local map tine = " << du_seg << std::endl;
		//update gaussian object
		//기존 마스크에 추가된 내용에 대해서만 갱신
		//prev와 curr이어야 함. raft는 안됨.

		didx = 6;
		DebugAssoSeg.Update(didx, DebugAssoSeg.Get(didx) + 1);

		UpdateGaussianObjectMap(pPairData->mapRaftSeg, pPrevSegMask, pCurrSegMask, InstanceType::SEG);

		didx = 7;
		DebugAssoSeg.Update(didx, DebugAssoSeg.Get(didx) + 1);

		/*std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
		auto du_u = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		auto du_o = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

		std::cout << "test = " << du_u << " " << du_o << std::endl;*/

		////리커버리 전략
		//from에서 연결 안된 영역
		//1) from rect안에서
		cv::Mat cimg = pNewBF->img.clone();
		cv::Mat pimg = pPrevBF->img.clone();

		std::vector<std::pair<cv::Point2f, cv::Point2f>> vecMatch, vecMatch2;
		
		cv::Mat resImage;
		SLAM->VisualizeMatchingImage(resImage, pimg, cimg, vecMatch, mapName, 0, cv::Scalar(0, 255, 255));
		VisualizeAssociation(SLAM, pPairData, mapName, 2);

		didx = 8;
		DebugAssoSeg.Update(didx, DebugAssoSeg.Get(didx) + 1);
				
		std::cout << "Debug test SEG = ";
		auto debug = DebugAssoSeg.Get();
		for (auto pair : debug)
		{
			std::cout << pair.first<<"="<<pair.second << "  ";
		}
		std::cout << std::endl;
	}

	void AssociationManager::AssociationWithSAM(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key
		, const std::string& mapName, const std::string& userName, const int _type
		, AssoFramePairData* pPairData) 
	{
		auto pPrevBF = pPairData->mpFrom;
		auto pNewBF = pPairData->mpTo;

		auto pCurrKF = pNewBF->mpRefKF;
		auto pPrevKF = pPrevBF->mpRefKF;

		auto pCurrSegMask = pNewBF->mapMasks.Get("yoloseg");
		auto pPrevSegMask = pPrevBF->mapMasks.Get("yoloseg");

		auto pCurrSegInstance = pCurrSegMask->FrameInstances.Get();
		auto pPrevSegInstance = pPrevSegMask->FrameInstances.Get();

		auto mapSamInstance = pPairData->mpSamIns->FrameInstances.Get();
		std::map<int, FrameInstance*> mapRaftInstance;

		for (auto pair : pPairData->mapReqRaft)
		{
			auto pid = pair.first;
			auto pIns = pPairData->mpRaftIns->FrameInstances.Get(pid);
			mapRaftInstance[pid] = pIns;
		}

		std::map<std::pair<int, int>, AssoMatchRes*> res, mapSuccess;
		std::map<int, int> mnMatchedIdx;
		CalculateIOU(mapRaftInstance, mapSamInstance, res);
		EvaluateMatchResults(res, mapSuccess);

		for (auto pair : mapSuccess)
		{
			auto pid = pair.first.first;
			auto sid = pair.first.second;

			auto ares = pair.second;

			if (ares->iou > 0.5)
			{
				ares->res = true;
				//샘 인스턴스가 중복될 수 있음. 확인 필요
				mnMatchedIdx[sid] = pid;
			}
		}

		for (auto pair : mapSamInstance)
		{
			auto sid = pair.first;
			auto pNewSAM = pair.second;

			//???????????????
			if (CheckAddNewInstance(pCurrSegInstance, pNewSAM) && mnMatchedIdx.count(sid))
			{
				int nNewID = AddNewInstance(pCurrSegMask, pNewSAM);
				int pid = mnMatchedIdx[sid];
				pCurrSegInstance[nNewID] = pNewSAM;

				auto ares = mapSuccess[std::make_pair(pid, sid)];
				ares->id2 = nNewID;

				pPairData->mapRaftSam[pid] = nNewID;

				//어소시에이션 결과 교체
				//pReqMask->mapResAssociation[pid] = ares;
				//pReqSegMask->mapResAssociation[pid] = ares;
				/*pReqMask->mapResAssociation[pid]->id2 = nNewID;
				pReqSegMask->mapResAssociation[pid]->id2 = nNewID;*/

				//pNewSamMask->FrameInstances.Update(nNewID, pNewSAM);
				//pNewSamMask->GaussianMaps.Update(nNewID, nullptr);
			}
			else {
				pPairData->setSamToFailed.insert(sid);
			}
		}

		for (auto pair : pPairData->mapReqRaft)
		{
			auto pid = pair.first;
			if (!pPairData->mapRaftSam.count(pid))
				pPairData->setSamFromFailed.insert(pid);
		}

		UpdateGaussianObjectMap(pPairData->mapRaftSam, pPrevSegMask, pCurrSegMask, InstanceType::SEG);
		VisualizeAssociation(SLAM, pPairData, mapName, 2, 1);

		//TestUncertainty(SLAM, ObjSLAM, pPairData);

	}
	void AssociationManager::AssociateLocalMapWithUncertainty(EdgeSLAM::SLAM* SLAM, AssoFramePairData* pPairData) {
		//오브젝트 집합
		////1. local map
		////2. 그리드 + 카메라에서 그리드 보이면(V)
		////3. 객체 끼리 연결
		//조건
		////1. prev frame에 해당 객체가 연결이 되지 않음
		////2. prev에 포함된 객체와 연결된 객체임
		
		/*int didx = 0;
		didx = 0;
		DebugAssoSAM.Update(didx, DebugAssoSAM.Get(didx) + 1);*/

		float chi = sqrt(5.991);

		auto pPrevBF = pPairData->mpFrom;
		auto pCurrBF = pPairData->mpTo;

		auto pPrevKF = pPrevBF->mpRefKF;
		auto pCurrKF = pCurrBF->mpRefKF;

		auto pPrevSegMask = pPrevBF->mapMasks.Get("yoloseg");
		auto pCurrSegMask = pCurrBF->mapMasks.Get("yoloseg");

		const cv::Mat Kc = pCurrKF->K.clone();
		const cv::Mat Tc = pCurrKF->GetPose();
		const cv::Mat Rc = Tc.rowRange(0, 3).colRange(0, 3);
		const cv::Mat tc = Tc.rowRange(0, 3).col(3);

		int w = pCurrKF->mpCamera->mnWidth;
		int h = pCurrKF->mpCamera->mnHeight;

		int margin = 10;
		int w2 = w - margin;
		int h2 = h - margin;

		std::map<int, GOMAP::GaussianObject*> mpGOs;
		GetLocalObjectMaps(pPrevSegMask, mpGOs); 

		//프로젝션
		std::map<int, FrameInstance*> mapFailedCurrIns;
		std::map<int, int> mapTempMatches;
		std::map<int, AssoMatchRes*> mapAssoRes;//id1 : 자기자신, id2 : 글로벌 객체 id
		std::map<int, std::set<GOMAP::GaussianObject*>> mapAssoGOs;

		cv::Mat cimg = pCurrBF->img.clone();

		for (auto cid : pPairData->setSegToFailed)
		{
			mapFailedCurrIns[cid] = pCurrSegMask->FrameInstances.Get(cid);
			auto ares = new AssoMatchRes();
			ares->id1 = cid;
			mapAssoRes[cid] = ares;
			cv::rectangle(cimg, mapFailedCurrIns[cid]->rect, cv::Scalar(0, 0, 255), 2);
		}

		for (auto pair : mpGOs)
		{
			auto pG = pair.second;
			auto map2D = pG->Project2D(Kc, Rc, tc);
			auto pt = cv::Point2f(map2D.center);
			if (pt.x < margin || pt.x >= w2 || pt.y < margin || pt.y >= h2)
				continue;
			
			//map region feature
			ObjectRegionFeatures morf(pG, nullptr);
			morf.map2D = morf.mpRefMap->Project2D(Kc, Rc, tc);
			morf.region = morf.map2D.CalcEllipse(chi);
			morf.rect = morf.region.boundingRect();
			morf.map2D.rect = morf.rect;

			cv::ellipse(cimg, morf.region, cv::Scalar(255, 0, 0), 2);
			cv::putText(cimg, std::to_string(pG->id), morf.region.center, 2, 1.3, cv::Scalar(255, 0, 0), 2);

			std::vector<int> vecTempIDXs;
			for (auto pair2 : mapFailedCurrIns)
			{
				auto cid = pair2.first;
				auto cins = pair2.second;

				float iou = morf.map2D.CalcIOU(cins->rect);
				if (iou > 0.0)
				{
					vecTempIDXs.push_back(cid);
					cv::ellipse(cimg, morf.region, cv::Scalar(0, 255, 0), 2);
				}
			}

			if (vecTempIDXs.size() == 0)
				continue;

			ExtractRegionFeatures(pCurrKF, morf.region, morf.keypoints, morf.mappoints, morf.descriptors);

			//obs frame region feature
			auto obs = pG->mObservations.Get();
			for (auto pair2 : obs)
			{
				auto pIns = pair2.second;
				auto pTempKF = pIns->mpRefKF;

				ObjectRegionFeatures porf(pG, pIns);
				porf.rect = porf.mpRefIns->rect;
				ExtractRegionFeatures(pTempKF, porf.rect, porf.keypoints, porf.mappoints, porf.descriptors);

				std::vector<std::pair<int, int>> vecMatches, vecFilteredMatches, vecResMatches;
				ObjectMatcher::SearchInstance(porf.descriptors, morf.descriptors, vecMatches);
				bool bNew = ObjectMatcher::removeOutliersWithMahalanobis(porf.keypoints, morf.keypoints, vecMatches, vecFilteredMatches);
				
				if (bNew) {
					//std::cout << "matching test = " << pG->id << " " << vecFilteredMatches.size() << " " << bNew << std::endl;
					
					auto pUins = GenerateFrameInsWithUncertainty(pCurrKF, porf, morf, vecFilteredMatches);

					for (auto cid : vecTempIDXs)
					{
						auto cins = mapFailedCurrIns[cid];
						float iou = CalculateIOU(pUins->mask, cins->mask, pUins->area, 1);
						auto ares = mapAssoRes[cid];
						if (iou > 0.5)
						{  
							ares->res = true;
							if (ares->iou < iou)
							{
								ares->iou = iou;
								ares->id2 = pG->id;
							}
							//pPairData->mapRaftSeg[pid] = pair2.first;
							if (!mapAssoGOs[cid].count(pG))
								mapAssoGOs[cid].insert(pG);
						}
					}
				}
			}

		}

		//뷰안 체크

		//오브젝트 끼리 오버랩

		//불확실성으로 리전 피쳐 생성
		
		//프레임 옵저베이션으로 프레임 영역 특징 생성
		////모든 프레임과 일단 비교하게 하고 성능이 느리면 수정이 필요함.

		//매칭  
		for (auto pair : mapAssoRes)
		{
			auto cid = pair.first;

			auto ares = pair.second;
			if (!ares->res)
				continue;

			auto spGOs = mapAssoGOs[cid];
			auto pG = mpGOs[ares->id2];
			if (spGOs.size() > 1)
			{
				std::cout << cid << " == " << spGOs.size() << std::endl;
			}
			cv::rectangle(cimg, mapFailedCurrIns[cid]->rect, cv::Scalar(0, 255, 0), 2);
			pCurrSegMask->GaussianMaps.Update(cid, pG);
		}
		/*std::cout << "Debug Local Map = ";
		didx = 1;
		DebugAssoSAM.Update(didx, DebugAssoSAM.Get(didx) + 1);

		auto debug = DebugAssoSAM.Get();
		for (auto pair : debug)
		{
			std::cout << pair.first << "=" << pair.second << "  ";
		}
		std::cout << std::endl;*/

		std::stringstream ss;
		ss.str("");
		ss << "../res/asso/" << pPairData->toid << "_" << pPairData->fromid << "_" << 2 << ".png";
		cv::imwrite(ss.str(), cimg);
	}

	void AssociationManager::TestUncertainty(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, AssoFramePairData* pPairData) {
		
		float chi = sqrt(5.991);
		
		auto pPrevBF = pPairData->mpFrom;
		auto pCurrBF = pPairData->mpTo;

		auto pPrevKF = pPrevBF->mpRefKF;
		auto pCurrKF = pCurrBF->mpRefKF;

		auto vpBFs = ObjSLAM->GetConnectedBoxFrames(pCurrKF, 5 );

		auto pCurrSegMask = pCurrBF->mapMasks.Get("yoloseg");

		const cv::Mat Kc = pCurrKF->K.clone();
		const cv::Mat Tc = pCurrKF->GetPose();
		const cv::Mat Rc = Tc.rowRange(0, 3).colRange(0, 3);
		const cv::Mat tc = Tc.rowRange(0, 3).col(3);

		int w = pCurrKF->mpCamera->mnWidth;
		int h = pCurrKF->mpCamera->mnHeight;

		//과거 프레임을 돌면서 현재 프레임에 프로젝션
		for (auto pBF : vpBFs)
		{
			auto pKF = pBF->mpRefKF;
			auto pSegMask = pBF->mapMasks.Get("yoloseg");
			auto mapTempGOs = pSegMask->GaussianMaps.Get();

			cv::Mat pimg = pBF->img.clone();
			cv::Mat cimg = pCurrBF->img.clone();

			std::vector<std::pair<cv::Point2f, cv::Point2f>> vecTestSuccess, vecTestFailed;
			std::vector<std::pair<cv::Point2f, cv::Point2f>> vecSuccess, vecFailed;

			for (auto pair : mapTempGOs)
			{
				auto pid = pair.first;
				auto pG = pair.second;
				if (!pG || !pG->mbInitialized)
					continue;
				auto pins = pSegMask->FrameInstances.Get(pid);
				ObjectRegionFeatures porf(nullptr, pins);
				porf.rect = porf.mpRefIns->rect;
				ExtractRegionFeatures(pPrevKF, porf.rect, porf.keypoints, porf.mappoints, porf.descriptors);

				ObjectRegionFeatures morf(pG, pins);
				morf.map2D = morf.mpRefMap->Project2D(Kc, Rc, tc);
				morf.region = morf.map2D.CalcEllipse(chi);
				morf.rect = morf.region.boundingRect();
				ExtractRegionFeatures(pCurrKF, morf.region, morf.keypoints, morf.mappoints, morf.descriptors);

				cv::rectangle(pimg, porf.rect, cv::Scalar(0, 255, 0), 1);
				cv::ellipse(cimg, morf.region, cv::Scalar(0, 255, 0), 1);

				//mp test
				for (int i = 0; i < porf.keypoints.size(); i++) {
					auto kp = porf.keypoints[i];
									
					auto mp = porf.mappoints[i];

					if (!mp || mp->isBad())
						continue;

					auto pt1 = kp.pt;
					

					if (mp->IsInKeyFrame(pCurrKF)) {
						auto idx2 = mp->GetIndexInKeyFrame(pCurrKF);
						auto pt2 = pCurrKF->mvKeys[idx2].pt;
						auto respair = std::make_pair(pt1, pt2);
						vecTestSuccess.push_back(respair);
					}
					else {
						auto pt2 = CommonUtils::Geometry::ProjectPoint(mp->GetWorldPos(), Kc, Rc, tc);
						auto respair = std::make_pair(pt1, pt2);
						vecTestFailed.push_back(respair);
					}
				}
			}
			//기록
			for (int i = 0; i < vecTestSuccess.size(); i += 5) {
				vecSuccess.push_back(vecTestSuccess[i]);
			}
			for (int i = 0; i < vecTestFailed.size(); i += 5) {
				vecFailed.push_back(vecTestFailed[i]);
			}
			cv::Mat resImage;
			SLAM->VisualizeMatchingImage(resImage, pimg, cimg, vecFailed, "testmap", -1, cv::Scalar(0, 0, 255));
			SLAM->VisualizeMatchingImage(resImage, vecSuccess, "testmap", -1, cv::Scalar(255, 0, 0));

			std::stringstream ss;
			ss.str("");
			ss << "../res/asso/test_" << pCurrKF->mnId << "_" << pKF->mnId << ".png";
			cv::imwrite(ss.str(), resImage);
		}

	}

	void AssociationManager::AssociationWithUncertainty(EdgeSLAM::SLAM* SLAM, AssoFramePairData* pPairData) {
		float chi = sqrt(5.991);

		auto pPrevBF = pPairData->mpFrom;
		auto pCurrBF = pPairData->mpTo;

		auto pPrevKF = pPrevBF->mpRefKF;
		auto pCurrKF = pCurrBF->mpRefKF;
		
		auto pPrevSegMask = pPrevBF->mapMasks.Get("yoloseg");
		auto pCurrSegMask = pCurrBF->mapMasks.Get("yoloseg");

		const cv::Mat Kc = pCurrKF->K.clone();
		const cv::Mat Tc = pCurrKF->GetPose();
		const cv::Mat Rc = Tc.rowRange(0, 3).colRange(0, 3);
		const cv::Mat tc = Tc.rowRange(0, 3).col(3);

		int w = pCurrKF->mpCamera->mnWidth;
		int h = pCurrKF->mpCamera->mnHeight;

		std::map<int, FrameInstance*> mapFailedCurrIns;
		for (auto cid : pPairData->setSegToFailed)
		{
			mapFailedCurrIns[cid] = pCurrSegMask->FrameInstances.Get(cid);
		}

		for (auto pid : pPairData->setSegFromFailed)
		{
			auto pG = pPrevSegMask->GaussianMaps.Get(pid);
			if (pG && pG->mbInitialized) {
				auto pins = pPrevSegMask->FrameInstances.Get(pid);
				ObjectRegionFeatures porf(nullptr, pins);
				porf.rect = porf.mpRefIns->rect;
				ExtractRegionFeatures(pPrevKF, porf.rect, porf.keypoints, porf.descriptors);

				ObjectRegionFeatures morf(pG, pins);
				morf.map2D = morf.mpRefMap->Project2D(Kc, Rc, tc);
				morf.region = morf.map2D.CalcEllipse(chi);
				morf.rect = morf.region.boundingRect();
				ExtractRegionFeatures(pCurrKF, morf.region, morf.keypoints, morf.descriptors);

				//두 region feaeture 비교
				std::vector<std::pair<int, int>> vecMatches, vecFilteredMatches, vecResMatches;
				ObjectMatcher::SearchInstance(porf.descriptors, morf.descriptors, vecMatches);
				bool bNew = ObjectMatcher::removeOutliersWithMahalanobis(porf.keypoints, morf.keypoints, vecMatches, vecFilteredMatches);

				//새로운 인스턴스 생성.
				//그 인스턴스를 새로운 프레임과 비교
				if (bNew)
				{
					auto pUins = GenerateFrameInsWithUncertainty(pCurrKF, porf, morf, vecFilteredMatches);

					for (auto pair2 : mapFailedCurrIns)
					{
						auto cins = pair2.second;
						float iou = CalculateIOU(pUins->mask, cins->mask, pUins->area, 1);
						if (iou > 0.5)
						{
							auto cid = pair2.first;
							pPairData->mapRaftSeg[pid] = cid;
							pPairData->setSegFromFailed.erase(pid);
							pPairData->setSegToFailed.erase(cid);
						}
					}
				}
			}
		}
	}

	FrameInstance* AssociationManager::GenerateFrameInsWithUncertainty(EdgeSLAM::KeyFrame* pKF, const ObjectRegionFeatures& prev, const ObjectRegionFeatures& map, const std::vector<std::pair<int, int>>& vecMatches)
	{

		std::vector<cv::Point2f> pointsA, pointsB;
		for (const auto& match : vecMatches) {
			pointsA.push_back(prev.keypoints[match.first].pt);
			pointsB.push_back(map.keypoints[match.second].pt);
		}
		cv::Scalar meanA = cv::mean(pointsA);
		cv::Scalar meanB = cv::mean(pointsB);

		auto rectA = prev.rect;
		cv::Point2f displacement(meanB[0] - meanA[0], meanB[1] - meanA[1]);
		
		auto pNewIns = prev.mpRefIns->ConvertedInstasnce(pKF, displacement);
		return pNewIns;

	}

	void AssociationManager::VisualizeAssociation(EdgeSLAM::SLAM* SLAM, AssoFramePairData* pPairData, std::string mapName, int num_vis, int type) {
		
		auto pPrevBF = pPairData->mpFrom;
		auto pNewBF = pPairData->mpTo;

		auto pPrevKF = pPrevBF->mpRefKF;
		auto pCurrKF = pNewBF->mpRefKF;

		cv::Mat pColorImg = pPrevBF->img.clone();
		cv::Mat cColorImg = pNewBF->img.clone();
		
		//인스턴스
		auto pCurrSeg = pNewBF->mapMasks.Get("yoloseg");
		auto pPrevSeg = pPrevBF->mapMasks.Get("yoloseg");

		auto mapCurrIns = pCurrSeg->FrameInstances.Get();
		auto mapPrevIns = pPrevSeg->FrameInstances.Get();

		std::map<int, FrameInstance*> mapSamIns;
		auto pSamMask = pPairData->mpSamIns;
		if (pSamMask)
			mapSamIns = pSamMask->FrameInstances.Get();

		cv::Scalar color1(255, 0, 0);
		cv::Scalar color2(0, 0, 255);
		cv::Scalar color3(0, 255, 0);

		for (auto pair : mapPrevIns)
		{
			int id = pair.first;
			auto p = pair.second;
			if (id == 0)
				continue;
			auto c = color1;
			int thick = 2;
			if (pPairData->setSegFromFailed.count(id)){
				c = color2;
				thick = 5;
			}
			if (pPairData->setSamFromFailed.count(id)){
				c = color3;
				thick = 5;
			}
			cv::rectangle(pColorImg, p->rect, c, thick);
		}
		for (auto pair : mapCurrIns)
		{
			int id = pair.first;
			auto p = pair.second;
			if (id == 0)
				continue;
			auto c = color1;
			int thick = 2;
			
			if (pPairData->setSegToFailed.count(id))
			{
				c = color2;
				thick = 5;
			}
			
			cv::rectangle(cColorImg, p->rect, c, thick);
		}
		for (auto pair : mapSamIns)
		{
			int id = pair.first;
			auto p = pair.second;
			if (id == 0)
				continue;
			auto c = color3;
			int thick = 10;

			if (!pPairData->setSamToFailed.count(id))
			{
				continue;
			}
			cv::rectangle(cColorImg, p->rect, c, thick);
		}

		//라프트 세그
		std::vector<std::pair<cv::Point2f, cv::Point2f>> vecSegMatch, vecSamMatch;
		for (auto pair : pPairData->mapRaftSeg)
		{
			auto pid = pair.first;
			auto cid = pair.second;

			auto pIns = mapPrevIns[pid];
			auto cIns = mapCurrIns[cid];

			auto pt1 = pIns->pt;
			auto pt2 = cIns->pt;
			vecSegMatch.push_back(std::make_pair(pt1, pt2));
		}

		//라프트 매칭
		for (auto pair : pPairData->mapRaftSam)
		{
			auto pid = pair.first;
			auto cid = pair.second;

			auto pIns = mapPrevIns[pid];
			auto cIns = mapCurrIns[cid];

			auto pt1 = pIns->pt;
			auto pt2 = cIns->pt;
			vecSamMatch.push_back(std::make_pair(pt1, pt2));
		}

		//맵 아이디 포함



		//Curr GO
		const cv::Mat Tp = pPrevKF->GetPose();
		const cv::Mat Kp = pPrevKF->K.clone();
		const cv::Mat Rp = Tp.rowRange(0, 3).colRange(0, 3);
		const cv::Mat tp = Tp.rowRange(0, 3).col(3);

		const cv::Mat Tc = pCurrKF->GetPose();
		const cv::Mat Kc = pCurrKF->K.clone();
		const cv::Mat Rc = Tc.rowRange(0, 3).colRange(0, 3);
		const cv::Mat tc = Tc.rowRange(0, 3).col(3);

		float chi = sqrt(5.991);
		auto mapPrevGO = pPrevSeg->GaussianMaps.Get();
		auto mapCurrGO = pCurrSeg->GaussianMaps.Get();
		
		for (auto pair : mapPrevGO)
		{
			auto pG = pair.second;
			if (!pG || !pG->mbInitialized)
				continue;
			auto pIns = mapPrevIns[pair.first];
			auto pt1 = pIns->pt;
			
			GOMAP::GO2D g = pG->Project2D(Kp, Rp, tp);
			g.rect = g.CalcRect();
			auto _e = g.CalcEllipse();
			auto pt2 = cv::Point(g.rect.x + g.rect.width / 2, g.rect.y + g.rect.height / 2);
			cv::line(pColorImg, pt1, pt2, cv::Scalar(255, 0, 0), 3);
			cv::rectangle(pColorImg, g.rect, cv::Scalar(255), 2);
			cv::ellipse(pColorImg, _e, cv::Scalar(255, 0, 255), -1);
			cv::putText(pColorImg, std::to_string(pG->id), _e.center, 2, 1.3, cv::Scalar(255, 0, 0), 2);
		}
		for (auto pair : mapCurrGO)
		{
			auto pIns = mapCurrIns[pair.first];
			auto pG = pair.second;
			if (!pG || !pG->mbInitialized){
				cv::rectangle(cColorImg, pIns->rect, cv::Scalar(0,0,0), 2);
				continue;
			}
			auto pt1 = pIns->pt;

			GOMAP::GO2D g = pG->Project2D(Kc, Rc, tc);
			g.rect = g.CalcRect();
			auto _e = g.CalcEllipse();
			auto pt2 = cv::Point(g.rect.x + g.rect.width / 2, g.rect.y + g.rect.height / 2);
			cv::line(cColorImg, pt1, pt2, cv::Scalar(255, 0, 0), 3);
			cv::rectangle(cColorImg, g.rect, cv::Scalar(255), 2);
			cv::ellipse(cColorImg, _e, cv::Scalar(255, 0, 255), -1);
			cv::putText(cColorImg, std::to_string(pG->id), _e.center, 2, 1.3, cv::Scalar(255, 0, 0), 2);
		}

		//시각화
		cv::Mat resImage;
		SLAM->VisualizeMatchingImage(resImage, pColorImg, cColorImg, vecSegMatch, mapName, num_vis, cv::Scalar(0,255,255));
		SLAM->VisualizeMatchingImage(resImage, vecSamMatch, mapName, num_vis, cv::Scalar(255,0,255));

		std::stringstream ss;
		ss.str("");
		ss << "../res/asso/" << pPairData->toid <<"_"<<pPairData->fromid << "_" << type << ".png";
		cv::imwrite(ss.str(), resImage);
	}

	/////
	void AssociationManager::ConvertMaskWithRAFT(const std::map<int, FrameInstance*>& mapSourceInstance, std::map<int, FrameInstance*>& mapRaftInstance
		, EdgeSLAM::KeyFrame* pRefKF, const cv::Mat& flow) {
		for (auto pair : mapSourceInstance)
		{
			int sid = pair.first;
			if (sid == 0)
				continue;
			auto p = pair.second;
			/*if (!p)
			{
				std::stringstream ss;
				ss << "prev::instance::error::" << pPrevSegMask->mnMaxId << " " << pPrevIns.size() << " " << pPrevSegMask->MapInstances.Size();
				ObjSLAM->vecObjectAssoRes.push_back(ss.str());
				continue;
			}*/
			//새로 테스트 하는 것
			auto pRaftIns = new FrameInstance(pRefKF, InstanceType::RAFT);
			if (InstanceSim::ComputeRaftInstance(flow, p, pRaftIns)) {
				mapRaftInstance[sid] = pRaftIns;
				pRaftIns->mpGO = p->mpGO;
			}
			else {
				delete pRaftIns;
			}
		}
	}
	float AssociationManager::CalculateIOU(const cv::Mat& mask1, const cv::Mat& mask2, float area1, int id) {
		cv::Mat overlap = mask1 & mask2;
		float nOverlap = (float)cv::countNonZero(overlap);
		float iou = 0.0;
		if (nOverlap == 0) {
			
		}
		else {
			if (id == 0) {
				iou = nOverlap / area1;
			}
			if (id > 0)
			{
				cv::Mat total = mask1 | mask2;
				float nUnion = (float)cv::countNonZero(total);
				iou = nOverlap / nUnion;
			}
		}
		return iou;

	}
	void AssociationManager::CalculateIOU(const std::map<int, FrameInstance*>& mapPrevInstance, const std::map<int, FrameInstance*>& mapCurrInstance
		, std::map<std::pair<int, int>, AssoMatchRes*>& mapIOU, float th)
	{
		for (auto pair1 : mapPrevInstance) {
			int id1 = pair1.first;
			if (id1 == 0)
				continue;

			const cv::Mat pmask = pair1.second->mask;
			float area1 = (float)cv::countNonZero(pmask);
			auto prevIns = pair1.second;

			//std::pair<int, float> bestMatch;
			//bestMatch.second = 0.0;

			for (auto pair2 : mapCurrInstance) {
				int id2 = pair2.first;

				auto key = std::make_pair(id1, id2);

				auto currIns = pair2.second;

				AssoMatchRes* ares = new AssoMatchRes(id1, id2, prevIns->type, currIns->type);
				
				const cv::Mat cmask = pair2.second->mask;

				//cv::Mat overlap = pmask & cmask;
				//float nOverlap = (float)cv::countNonZero(overlap);

				////겹치는게 없으면 무시
				//if (nOverlap == 0) {
				//	ares->iou = 0.0;
				//}
				//else {
				//	float iou = 0.0;

				//	//백그라운드와 비교인지, 추정된 인스턴스와 비교인지에 따라서 다름.
				//	//백그라운드와 비교시 SAM 요청 가능
				//	if (id2 == 0) {
				//		iou = nOverlap / area1;
				//	}
				//	if (id2 > 0)
				//	{
				//		cv::Mat total = pmask | cmask;
				//		float nUnion = (float)cv::countNonZero(total);
				//		iou = nOverlap / nUnion;
				//	}
				//	ares->iou = iou;
				//	
				//	/*if (iou > bestMatch.second)
				//	{
				//		bestMatch.first = id2;
				//		bestMatch.second = iou;
				//	}*/
				//}
				ares->iou = CalculateIOU(pmask, cmask, area1, id2);
				mapIOU[key] = ares;
			}

		}
	}
	
	void AssociationManager::EvaluateMatchResults(std::map<std::pair<int, int>, AssoMatchRes*>& mapIOU
		, std::map<std::pair<int, int>, AssoMatchRes*>& mapSuccess, float th) {

		/*std::vector<int> vMatchedDistance(N2, INT_MAX);
		std::vector<int> vMatched1(N1, -1);
		std::vector<int> vMatched2(N2, -1);*/

		std::map<int, float> vMatchedDisatnce1, vMatchedDisatnce2;
		std::map<int, int> vMatched1, vMatched2;

		std::vector < std::pair<float, std::pair<int, int>>> vPairs;

		//prev에서 중복을 제거하고
		//그걸 바탕으로 curr에서 매칭을 수행.

		for (auto pair : mapIOU)
		{
			auto key = pair.first;
			auto ares = pair.second;

			auto pid = key.first;
			auto cid = key.second;

			float val = ares->iou;

			if (val > th)
			{
				vPairs.push_back(std::make_pair(val, key));
				/*if (cid == 0)
				{
					if (val < vMatchedDisatnce1[pid])
						continue;
					vMatchedDisatnce1[pid] = val;
					vMatched1[pid] = cid;
				}
				else {
					
					if (val < vMatchedDisatnce2[cid])
						continue;
					vMatchedDisatnce2[cid] = val;
					auto prev = vMatched2[cid];
					if(prev > 0)
						vMatched1[prev] = -1;
					vMatched2[cid] = pid;
					vMatched1[pid] = cid;
				}*/
				vMatched1[pid] = -1;
				vMatched2[cid] = -1;
			}
		}
		std::sort(vPairs.begin(), vPairs.end(), std::greater<>());

		for (auto pair : vPairs)
		{
			float val = pair.first;
			auto key = pair.second;
			auto pid = key.first;
			auto cid = key.second;

			if (cid == 0)
			{
				if (vMatched1[pid] == -1)
				{
					vMatched1[pid] = cid;
				}
			}
			else {
				if (vMatched1[pid] == -1 && vMatched2[cid] == -1)
				{
					vMatched1[pid] = cid;
					vMatched2[cid] = pid;
				}
			}
		}

		for (auto pair : vMatched1) {
			int i1 = pair.first;
			int i2 = pair.second;
			if (i1 >= 0 && i2 >= 0)
			{
				mapSuccess[pair] = mapIOU[pair];
			}
		}
	}

	void AssociationManager::GetLocalObjectMaps(InstanceMask* pMask, std::map<int,GOMAP::GaussianObject*>& mpGOs) {
		auto mapPrevGOs = pMask->GaussianMaps.Get();
		std::set<InstanceMask*> setFrames;
		std::set<GOMAP::GaussianObject*> spGOs;
		//이전 프레임 객체 정보로부터 인접한 프레임 정보 획득
		for (auto pair : mapPrevGOs)
		{
			auto pG = pair.second;
			if (!pG || !pG->mbInitialized)
				continue;
			//mpGOs[pG->id] = pG;
			spGOs.insert(pG);
			auto mObs = pG->GetObservations();
			for (auto pair : mObs)
			{
				auto f = pair.first;
				if (f == pMask)
					continue;
				if (!setFrames.count(f))
					setFrames.insert(f);
			}
		}
		for (auto f : setFrames)
		{
			auto mapGOs = f->GaussianMaps.Get();
			for (auto pair : mapGOs)
			{
				auto pG = pair.second;
				if (!pG || !pG->mbInitialized)
					continue;
				if (spGOs.count(pG))
					continue;
				if (!mpGOs.count(pG->id)) {
					mpGOs[pG->id] = pG;
				}
			}
		}
	}
	void AssociationManager::GetObjectMap2Ds(const std::map<int, GOMAP::GaussianObject*>& mpGOs, BoxFrame* pBF, std::map<int, GOMAP::GO2D>& spG2Ds)
	{
		////프로젝션
		auto pCurrKF = pBF->mpRefKF;
		const cv::Mat Kc = pCurrKF->K.clone();
		const cv::Mat Tc = pCurrKF->GetPose();
		const cv::Mat Rc = Tc.rowRange(0, 3).colRange(0, 3);
		const cv::Mat tc = Tc.rowRange(0, 3).col(3);

		int w = pCurrKF->mpCamera->mnWidth;
		int h = pCurrKF->mpCamera->mnHeight;

		for (auto pair : mpGOs)
		{
			auto pG = pair.second;
			GOMAP::GO2D g = pG->Project2D(Kc, Rc, tc);
			g.rect = g.CalcRect();
			spG2Ds[pair.first] = g;
		}
	}

	void AssociationManager::ConvertFrameInstances(const std::map<int, GOMAP::GO2D>& spG2Ds, BoxFrame* pBF, std::map<int, FrameInstance*>& mapIns)
	{

		auto pKF = pBF->mpRefKF;
		int w = pKF->mpCamera->mnWidth;
		int h = pKF->mpCamera->mnHeight;

		for (auto pair : spG2Ds)
		{
			int id = pair.first;
			auto g2d = pair.second;

			auto pNewIns = new FrameInstance(pKF, InstanceType::MAP);
			//mask, rea, rect, contour, pt
			pNewIns->mask = cv::Mat::zeros(h, w, CV_8UC1); //ellipse로 교체하던가
			cv::rectangle(pNewIns->mask, g2d.rect, cv::Scalar(255, 255, 255), -1);
			pNewIns->pt = cv::Point2f(g2d.rect.x + g2d.rect.width / 2, g2d.rect.y + g2d.rect.height / 2);
			pNewIns->rect = g2d.rect;
			pNewIns->area = (float)g2d.rect.area();

			ConvertContour(g2d.rect, pNewIns->contour);
			mapIns[id] = pNewIns;
		}
	}

	void AssociationManager::ProjectObjectMap(InstanceMask* pCurrSeg, InstanceMask* pPrevSeg
		, BoxFrame* pNewBF
		, std::map<GOMAP::GaussianObject*, FrameInstance*>& mapInstance
	){
		//객체 맵 생성을 최소화하는 것
		//여러 곳에서 엉뚱한 위치에 샘을 요청하게 하는 것을 막는 것
		//객체를 바로 연결하도록 변경
		std::set<GOMAP::GaussianObject*> setGOs;
		std::set<InstanceMask*> setFrames;

		////키프레임에서 후보군 생성
		auto mapPrevGOs = pPrevSeg->GaussianMaps.Get();
		//이전 프레임 객체 정보로부터 인접한 프레임 정보 획득
		for (auto pair : mapPrevGOs)
		{
			auto pG  = pair.second;
			if (!pG)
				continue;
			setGOs.insert(pG);
			auto mObs = pG->GetObservations();
			for (auto pair : mObs)
			{
				auto f = pair.first;
				if (f == pCurrSeg || f == pPrevSeg)
					continue;
				if (!setFrames.count(f))
					setFrames.insert(f);
			}
		}

		//인접 프레임으로부터 객체 정보 추가
		for (auto f : setFrames)
		{
			auto mapGOs = f->GaussianMaps.Get();
			for (auto pair : mapGOs)
			{
				auto pG = pair.second;
				if (!pG)
					continue;
				if (setGOs.count(pG))
					continue;
				setGOs.insert(pG);
			}
		}
		////키프레임에서 후보군 생성

		////프로젝션
		auto pCurrKF = pNewBF->mpRefKF;
		const cv::Mat Kc = pCurrKF->K.clone();
		const cv::Mat Tc = pCurrKF->GetPose();
		const cv::Mat Rc = Tc.rowRange(0, 3).colRange(0, 3);
		const cv::Mat tc = Tc.rowRange(0, 3).col(3);

		int w = pCurrKF->mpCamera->mnWidth;
		int h = pCurrKF->mpCamera->mnHeight;

		std::map<GOMAP::GaussianObject*,GOMAP::GO2D> map2DGO;
		for (auto pG : setGOs)
		{
			GOMAP::GO2D g = pG->Project2D(Kc, Rc, tc);
			g.rect = g.CalcRect();
			map2DGO[pG] = g;
		}
		////프로젝션

		////중복 체크
		for (auto pair1 : map2DGO)
		{
			auto p3d1 = pair1.first;
			auto p2d1 = pair1.second;

			for (auto pair2 : map2DGO)
			{
				auto p3d2 = pair2.first;
				auto p2d2 = pair2.second;

				if (p3d1 == p3d2)
					continue;

			}
		}

		//FrameInstance 추가
		//이전 프레임에 인스턴스와 연결되고, RAFT로 변환이 되면 해당 인스턴스는 윤곽정보로 변환됨.
		for (auto pG : setGOs)
		{
			auto g2d = map2DGO[pG];
			auto pNewIns = new FrameInstance(pCurrKF, InstanceType::MAP);
			//mask, rea, rect, contour, pt
			pNewIns->mask = cv::Mat::zeros(h, w, CV_8UC1);
			cv::rectangle(pNewIns->mask, g2d.rect, cv::Scalar(255, 255, 255), -1);
			pNewIns->pt = cv::Point2f(g2d.rect.x + g2d.rect.width / 2, g2d.rect.y + g2d.rect.height / 2);
			pNewIns->rect = g2d.rect;
			pNewIns->area = (float)g2d.rect.area();

			ConvertContour(g2d.rect, pNewIns->contour);
			mapInstance[pG] = pNewIns;
			pNewIns->mpGO = pG;
		}
		//FrameInstance 추가


		////2차원, 3차원 머지

		//A : 현재 프레임에 연결된 애들 제외

		//B : 이전 프레임에 연결된 애들 중 샘 요청한 애들 제외

		//C : 프로젝션 후 이미지 영역 바깥에 위치한 오브젝트 제외
		////뷰 안에 존재하는 비율도 체크

		//뎁스 정렬

		//A,B,C를 포함해서 살아남은 객체들 정렬

		////가림 확인
		//2d iou로 겹침 확인
		//3d 마할라노비스 거리로 결합 체크

		//프레임 인스턴스로 변환, 프레임의 id는 무시해도 됨.
	}

	void AssociationManager::RequestSAM(BoxFrame* pNewBF, std::map<int, FrameInstance*>& mapMaskInstance, 
		const std::map<GlobalInstance*, cv::Rect>& mapGlobalRect, 
		const std::vector<AssoMatchRes*>& vecResAsso, 
		int id1, int id2, const std::string& userName) {

		//전반적인 수정 필요.
		//오브젝트 맵을 인스턴스화 시킴

		//InstanceMask* pSamMask = nullptr;
		//if (!pNewBF->mapMasks.Count("missing"))
		//{
		//	pSamMask = new InstanceMask();
		//	pSamMask->id1 = id1; // previous frame
		//	pSamMask->id2 = id2; //current frame
		//	pNewBF->mapMasks.Update("missing", pSamMask);
		//}
		//else
		//{
		//	pSamMask = pNewBF->mapMasks.Get("missing");
		//}

		//cv::Mat ptdata(0, 1, CV_32FC1);
		//for (auto assores : vecResAsso)
		//{
		//	if (assores->res || !assores->req)
		//		continue;
		//	if (assores->nDataType != 1)
		//		continue;
		//	int mid = assores->id1;

		//	if (!mapMaskInstance.count(mid))
		//		std::cout << "raft error" << std::endl;
		//	auto pRaftIns = mapMaskInstance[mid];
		//	if (!pRaftIns)
		//	{
		//		//에러 출력
		//		/*std::stringstream ss;
		//		ss << "reqsam2,err,raftins,nullptr";
		//		ObjSystem->vecObjectAssoRes.push_back(ss.str());*/
		//		continue;
		//	}
		//	pSamMask->FrameInstances.Update(mid, pRaftIns);

		//	auto rect = pRaftIns->rect;
		//	cv::Mat temp = cv::Mat::zeros(4, 1, CV_32FC1);
		//	temp.at<float>(0) = rect.x;
		//	temp.at<float>(1) = rect.y;
		//	temp.at<float>(2) = rect.x + rect.width;
		//	temp.at<float>(3) = rect.y + rect.height;
		//	ptdata.push_back(temp);
		//}

		////글로벌 오브젝트 처리 필요
		//for (const auto pair : mapGlobalRect)
		//{
		//	auto rect = pair.second;
		//	cv::Mat temp = cv::Mat::zeros(4, 1, CV_32FC1);
		//	temp.at<float>(0) = rect.x;
		//	temp.at<float>(1) = rect.y;
		//	temp.at<float>(2) = rect.x + rect.width;
		//	temp.at<float>(3) = rect.y + rect.height;
		//	ptdata.push_back(temp);
		//}

		//if (ptdata.rows > 0) {
		//	//reqest
		//	int nobj = ptdata.rows;
		//	ptdata.push_back(cv::Mat::zeros(1500 - nobj, 1, CV_32FC1));
		//	std::string tsrc = userName + ".Image." + std::to_string(nobj);
		//	auto sam2key = "reqsam2";
		//	std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();
		//	auto du_upload = Utils::SendData(sam2key, tsrc, ptdata, id2, 15, t_start.time_since_epoch().count());
		//}

	}

	//업데이트는 분리
	void AssociationManager::UpdateGaussianObjectMap(std::map<int, int>& mapRes, InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, const InstanceType& type)
	{
		/*long long total = 0;
		long long total2 = 0;
		int n = 0;*/

		for (auto pair : mapRes)
		{
			auto pid = pair.first;
			auto cid = pair.second;

			auto pIns = pPrevSegMask->FrameInstances.Get(pid);
			auto cIns = pCurrSegMask->FrameInstances.Get(cid);

			auto pG1 = pPrevSegMask->GaussianMaps.Get(pid);
			auto pG2 = pCurrSegMask->GaussianMaps.Get(cid);

			//둘이 다르면 머지
			
			//둘중 하나가 있으면 하나 추가
			//아예 없으면 기본 생성
			//위치 초기화 필요하면 초기화

			GOMAP::GaussianObject* pGO = nullptr;
			if (!pG1 && !pG2)
			{
				pGO = new GOMAP::GaussianObject();
				pGO->AddObservation(pPrevSegMask, pIns);
				pGO->AddObservation(pCurrSegMask, cIns);

			}else if (pG1 && !pG2)
			{
				pG1->AddObservation(pCurrSegMask, cIns);
				if (pG1->mbInitialized) {
					GOMAP::Optimizer::ObjectOptimizer::ObjectPosOptimization(pG1);
					GaussianMapManager::UpdateObjectWithIncremental(pG1, cIns);
				}
				pGO = pG1;
			}else if (pG2 && !pG1)
			{
				pG2->AddObservation(pPrevSegMask, pIns);
				if (pG2->mbInitialized) {
					GOMAP::Optimizer::ObjectOptimizer::ObjectPosOptimization(pG2);
					GaussianMapManager::UpdateObjectWithIncremental(pG2, pIns);
				}
				pGO = pG2;
			}
			else if (pG1 && pG2)
			{
				if (pG1 != pG2)
				{
					std::cout << "update go error : 객체1, 객체2가 있는데 둘이 다름!!! 결합 필요" << std::endl;
				}
			}
			else {
				std::cout << "Update GO Map??????" << std::endl;
			}

			if (pGO && !pGO->mbInitialized)
			{
				bool b = GaussianMapManager::InitializeObject(pGO);

				if (pGO->mbInitialized)
					GOMAP::Optimizer::ObjectOptimizer::ObjectPosOptimization(pGO);
			}

			/*if (pG2)
			{
				std::cout << "already associated MAP?? Update GO Error" << std::endl;
			}
			GOMAP::GaussianObject* pGO = nullptr;
			if (type == InstanceType::SEG && !pG1 && !pG2)
			{
				pGO = GaussianMapManager::InitializeObject(pIns, cIns);
				pGO->AddObservation(pPrevSegMask, pIns);
				pGO->AddObservation(pCurrSegMask, cIns);
				if(pGO->mbInitialized)
					GOMAP::Optimizer::ObjectOptimizer::ObjectPosOptimization(pGO);
			}
			if (pG1 && !pG2)
			{
				if (pGO->mbInitialized){
					GOMAP::Optimizer::ObjectOptimizer::ObjectPosOptimization(pG1);
					GaussianMapManager::UpdateObjectWithIncremental(pG1, cIns);
				}
				pG1->AddObservation(pCurrSegMask, cIns);
				pGO = pG1;
			}*/

			////error check
			//if (!pG1 && pG2)
			//{
			//	std::cout << "update go error : 객체2만 있음" << std::endl;
			//}
			//if (pG1 && pG2) {
			//	if (pG1 == pG2)
			//	{

			//	}
			//	
			//}
			////error check

			//프레임 갱신
			//type == InstanceType::SEG && 
			if (pGO && !pG1)
			{
				pPrevSegMask->GaussianMaps.Update(pid, pGO);
			}
			if (pGO && !pG2)
			{
				pCurrSegMask->GaussianMaps.Update(cid, pGO);
			}

		}

	}
	void AssociationManager::UpdateGaussianObjectMap(
		InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, const InstanceType& type)
	{
		for (auto pair : pPrevSegMask->mapResAssociation)
		{
			auto pid = pair.first;
			auto ares = pair.second;
			auto cid = ares->id2;

			if (pid != ares->id1)
			{
				std::cout << "id error = Update GO Error" << std::endl;
			}

			auto pIns = pPrevSegMask->FrameInstances.Get(pid);
			auto cIns = pCurrSegMask->FrameInstances.Get(cid);

			auto pG1 = pPrevSegMask->GaussianMaps.Get(pid);
			auto pG2 = pCurrSegMask->GaussianMaps.Get(cid);
			if (pG2)
			{
				std::cout << "already associated MAP?? Update GO Error" << std::endl;
			}
			GOMAP::GaussianObject* pGO = nullptr;
			if (type == InstanceType::SEG && !pG1 && !pG2)
			{
				pGO = GaussianMapManager::InitializeObject(pIns, cIns);
				pGO->AddObservation(pPrevSegMask, pIns);
				pGO->AddObservation(pCurrSegMask, cIns);
				GOMAP::Optimizer::ObjectOptimizer::ObjectPosOptimization(pGO);
			}
			if (pG1 && !pG2)
			{
				GOMAP::Optimizer::ObjectOptimizer::ObjectPosOptimization(pG1);
				GaussianMapManager::UpdateObjectWithIncremental(pG1, cIns);
				pG1->AddObservation(pCurrSegMask, cIns);
				pGO = pG1;
			}
			if (!pG1 && pG2)
			{
				std::cout << "update go error : 객체2만 있음" << std::endl;
			}
			if (pG1 && pG2) {
				if (pG1 == pG2)
				{

				}
				if (pG1 != pG2)
				{
					std::cout<< "update go error : 객체1, 객체2가 있는데 둘이 다름" << std::endl;
				}
			}

			//프레임 갱신
			if (type == InstanceType::SEG && pGO && !pG1)
			{
				pPrevSegMask->GaussianMaps.Update(pid, pGO);
			}
			if (pGO && !pG2)
			{
				pCurrSegMask->GaussianMaps.Update(cid, pGO);
			}
		}
	}
	void AssociationManager::UpdateGaussianObjectMap(
		std::map<std::pair<int, int>, GOMAP::GaussianObject*>& mapGO,
		InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, const InstanceType& type) {

		for (auto pair : mapGO)
		{
			int pid = pair.first.first;
			int cid = pair.first.second;

			auto pIns = pPrevSegMask->FrameInstances.Get(pid);
			auto cIns = pCurrSegMask->FrameInstances.Get(cid);

			auto pG1 = pPrevSegMask->GaussianMaps.Get(pid);
			auto pG2 = pCurrSegMask->GaussianMaps.Get(cid);

			GOMAP::GaussianObject* pGO = nullptr;
			if (type == InstanceType::SEG && !pG1 && !pG2)
			{
				pGO = GaussianMapManager::InitializeObject(pIns, cIns);
				pGO->AddObservation(pPrevSegMask, pIns);
				pGO->AddObservation(pCurrSegMask, cIns);
				GOMAP::Optimizer::ObjectOptimizer::ObjectPosOptimization(pGO);
			}
			if (pG1 && !pG2)
			{
				GOMAP::Optimizer::ObjectOptimizer::ObjectPosOptimization(pG1);
				GaussianMapManager::UpdateObjectWithIncremental(pG1, cIns);
				pG1->AddObservation(pCurrSegMask, cIns);
				pGO = pG1;
			}

			//프레임 갱신
			if (type == InstanceType::SEG && pGO && !pG1)
			{
				pPrevSegMask->GaussianMaps.Update(pid, pGO);
				//pGO->AddObservation(pPrevSegMask, pIns);
			}
			if (pGO && !pG2)
			{
				pCurrSegMask->GaussianMaps.Update(cid, pGO);
				//pGO->AddObservation(pCurrSegMask, cIns);
			}
			/*if (!pGO)
			{
				std::cout << "???????????????" << pG1<< " " << pG2<< std::endl;
			}*/
			mapGO[pair.first] = pGO;
		}
	}
	void AssociationManager::VisualizeInstance(const std::map<int, GOMAP::GaussianObject*>& pMAP
		, std::map<int, FrameInstance*>& pPrev, const std::map<int, FrameInstance*>& pCurr, cv::Mat& vimg)
	{
		for (auto pair : pMAP)
		{
			auto pGO = pair.second;
			if (!pGO)
				continue;
			auto pIns =   pPrev[pair.first];
			if (!pIns)
				continue;
			cv::putText(vimg, std::to_string(pGO->id), pIns->pt, 2, 1.3, cv::Scalar(0, 0, 255), 2);
		}
		VisualizeInstance(pPrev, pCurr, vimg);
	}
	void AssociationManager::VisualizeInstance(const std::map<int, FrameInstance*>& pPrev, const std::map<int, FrameInstance*>& pCurr, cv::Mat& vimg) 
	{
		for (auto pair : pPrev)
		{
			auto pIns = pair.second;
			if (!pIns)
				continue;
			cv::rectangle(vimg, pIns->rect, cv::Scalar(255, 0, 0), 2);
		}
		for (auto pair : pCurr)
		{
			auto pIns = pair.second;
			if (!pIns)
				continue;
			cv::rectangle(vimg, pIns->rect, cv::Scalar(255, 255, 0), 2);
		}
	}

	void AssociationManager::VisualizeObjectMap(cv::Mat& vimg, InstanceMask* pMask, BoxFrame* pBF)
	{
		auto pCurrKF = pBF->mpRefKF;
		const cv::Mat Kc = pCurrKF->K.clone();
		const cv::Mat Tc = pCurrKF->GetPose();
		const cv::Mat Rc = Tc.rowRange(0, 3).colRange(0, 3);
		const cv::Mat tc = Tc.rowRange(0, 3).col(3);

		int w = pCurrKF->mpCamera->mnWidth;
		int h = pCurrKF->mpCamera->mnHeight;

		auto mpGOs = pMask->GaussianMaps.Get();
		auto mpIns = pMask->FrameInstances.Get();
		for (auto pair : mpGOs)
		{
			auto pIns = mpIns[pair.first];
			auto pG = pair.second;
			GaussianVisualizer::visualize2D(vimg, pG, Kc, Rc, tc, cv::Scalar(255, 0, 255), 1.0, -1);
			
		}
	}

	void AssociationManager::VisualizeMatchAssociation(EdgeSLAM::SLAM* SLAM, BoxFrame* pNewBF, BoxFrame* pPrevBF
		, InstanceMask* pCurrSeg, std::map<int, std::set<FrameInstance*>>& mapRes, std::string mapName, int num_vis)
	{
		auto pPrevKF = pPrevBF->mpRefKF;
		auto pCurrKF = pNewBF->mpRefKF;

		cv::Mat pColorImg = pPrevBF->img.clone();
		cv::Mat cColorImg = pNewBF->img.clone();

		auto mapCurrMasks = pCurrSeg->FrameInstances.Get();
		std::vector<std::pair<cv::Point2f, cv::Point2f>> vecPairVisualizedMatches;
		for (auto pair : mapRes)
		{
			int cid = pair.first;
			if (cid == 0)
				continue;
			auto pCurrIns = mapCurrMasks[cid];
			auto pt1 = pCurrIns->pt;
			cv::rectangle(cColorImg, pCurrIns->rect, cv::Scalar(255, 255, 0), 2);
			for (auto pPrevIns : pair.second)
			{
				cv::rectangle(pColorImg, pPrevIns->rect, cv::Scalar(255, 255, 0), 2);
				auto pt2 = pPrevIns->pt;
				vecPairVisualizedMatches.push_back(std::make_pair(pt2, pt1));
			}
		}
		//시각화
		cv::Mat resImage;
		SLAM->VisualizeMatchingImage(resImage, pColorImg, cColorImg, vecPairVisualizedMatches, mapName, num_vis);
	}

	void AssociationManager:: VisualizeAssociation(
		EdgeSLAM::SLAM* SLAM, 
		BoxFrame* pNewBF, BoxFrame* pPrevBF, 
		InstanceMask* pCurrSeg, InstanceMask* pPrevSeg, std::string mapName, int num_vis) {
				
		//segmentation + sam
		//color
		////prev
		////curr
		//mask
		//prev
		//curr
		
		int fid = pNewBF->mnId;

		auto pPrevKF = pPrevBF->mpRefKF;
		auto pCurrKF = pNewBF->mpRefKF;

		cv::Mat pColorImg = pPrevBF->img.clone();
		cv::Mat cColorImg = pNewBF->img.clone();

		//test code
		std::set<int> testPrevID, testCurrID;
		cv::Scalar color1 = cv::Scalar(255, 255, 0);
		cv::Scalar color2 = cv::Scalar(255, 0, 255);

		

		//prev gaussian map
		std::map<int, GOMAP::GO2D> mapPrevGO2D, mapCurrGO2D;
		const cv::Mat Tp = pPrevKF->GetPose();
		const cv::Mat Kp = pPrevKF->K.clone();
		const cv::Mat Rp = Tp.rowRange(0, 3).colRange(0, 3);
		const cv::Mat tp = Tp.rowRange(0, 3).col(3);
		auto setPrevGOs = pPrevSeg->GaussianMaps.Get();
		
		for (auto pair : setPrevGOs)
		{
			auto pG = pair.second;
			if (!pG){
				continue;
			}
			testPrevID.insert(pair.first);
			GOMAP::GO2D g = pG->Project2D(Kp, Rp, tp);
			g.rect = g.CalcRect();
			
			cv::rectangle(pColorImg, g.rect, cv::Scalar(0, 255, 255), 2);
			GaussianVisualizer::visualize2D(pColorImg, pG, Kp, Rp, tp, cv::Scalar(255, 0, 255),  1.0, -1);
		}
		//SLAM->VisualizeImage(mapName, pColorImg, 0);

		//curr gaussian map
		const cv::Mat Tc = pCurrKF->GetPose();
		const cv::Mat Kc = pCurrKF->K.clone();
		const cv::Mat Rc = Tc.rowRange(0, 3).colRange(0, 3);
		const cv::Mat tc = Tc.rowRange(0, 3).col(3);
		auto setCurrGOs = pCurrSeg->GaussianMaps.Get();
		for (auto pair : setCurrGOs)
		{
			auto pG = pair.second;
			if (!pG)
				continue;
			testCurrID.insert(pair.first);
			GaussianVisualizer::visualize2D(cColorImg, pG, Kc, Rc, tc, cv::Scalar(255, 0, 255), 1.0, -1);
		}
		//SLAM->VisualizeImage(mapName, cColorImg, 1);

		//prev mask
		auto mapPrevMasks = pPrevSeg->FrameInstances.Get();

		for (auto pair : mapPrevMasks)
		{
			auto pF = pair.second;
			if (!pF)
				continue;
			cv::Scalar color;
			if (testPrevID.count(pair.first))
				color = color1;
			else
				color = color2;
			for (int i = 0, j = 1; i < pF->contour.size(); i++, j++) {
				if (j == pF->contour.size())
					j = 0;
				cv::line(pColorImg, pF->contour[i], pF->contour[j], color, 2);
			}
		}

		//curr mask
		auto mapCurrMasks = pCurrSeg->FrameInstances.Get();
		for (auto pair : mapCurrMasks)
		{
			auto pF = pair.second;
			if (!pF)
				continue;
			cv::Scalar color;
			if (testCurrID.count(pair.first))
				color = color1;
			else
				color = color2;
			for (int i = 0, j = 1; i < pF->contour.size(); i++, j++) {
				if (j == pF->contour.size())
					j = 0;
				cv::line(cColorImg, pF->contour[i], pF->contour[j], color, 2);
			}
		}

		//association
		auto vecResAsso = pCurrSeg->mvResAsso.get();
		//cv::Mat pMask = cv::Mat::zeros(pColorImg.size(), CV_8UC1);
		//cv::Mat cMask = cv::Mat::zeros(cColorImg.size(), CV_8UC1);
		std::vector<std::pair<cv::Point2f, cv::Point2f>> vecPairVisualizedMatches;
		for (auto res : vecResAsso)
		{
			if (!res->res)
				continue;
			int id1 = res->id1;
			int id2 = res->id2;

			auto pt1 = mapPrevMasks[id1]->pt;
			auto pt2 = mapCurrMasks[id2]->pt;
			vecPairVisualizedMatches.push_back(std::make_pair(pt1, pt2));
		}
		
		//시각화
		cv::Mat resImage;
		SLAM->VisualizeMatchingImage(resImage, pColorImg, cColorImg, vecPairVisualizedMatches, mapName, num_vis);
	}

	void AssociationManager::VisualizeAssociation(EdgeSLAM::SLAM* SLAM
		, BoxFrame* pNewBF, BoxFrame* pPrevBF
		, InstanceMask* pCurrSeg
		, std::map<int, std::set<FrameInstance*>>& mapRes, std::string mapName, int num_vis)
	{
		int fid = pNewBF->mnId;

		auto pPrevKF = pPrevBF->mpRefKF;
		auto pCurrKF = pNewBF->mpRefKF;

		cv::Mat pColorImg = pPrevBF->img.clone();
		cv::Mat cColorImg = pNewBF->img.clone();

		//test code
		std::set<int> testPrevID, testCurrID;
		cv::Scalar color1 = cv::Scalar(255, 255, 0);
		cv::Scalar color2 = cv::Scalar(255, 0, 255);
		
		//frame instance matching
		auto mapCurrMasks = pCurrSeg->FrameInstances.Get();
		std::vector<std::pair<cv::Point2f, cv::Point2f>> vecPairVisualizedMatches;
		for (auto pair : mapRes)
		{
			int cid = pair.first;
			if (cid == 0)
				continue;
			auto pCurrIns = mapCurrMasks[cid];
			auto pt1 = pCurrIns->pt;
			cv::rectangle(cColorImg, pCurrIns->rect, cv::Scalar(255, 255, 0), 2);
			for (auto pPrevIns : pair.second)
			{
				cv::rectangle(pColorImg, pPrevIns->rect, cv::Scalar(255, 255, 0), 2);
				auto pt2 = pPrevIns->pt;
				vecPairVisualizedMatches.push_back(std::make_pair(pt2, pt1));
			}
		}

		//Curr GO
		const cv::Mat Tc = pCurrKF->GetPose();
		const cv::Mat Kc = pCurrKF->K.clone();
		const cv::Mat Rc = Tc.rowRange(0, 3).colRange(0, 3);
		const cv::Mat tc = Tc.rowRange(0, 3).col(3);
		auto setCurrGOs = pCurrSeg->GaussianMaps.Get();
		float chi = sqrt(5.991);
		for (auto pair : setCurrGOs)
		{
			auto pG = pair.second;
			if (!pG)
				continue;
			auto pCurrIns = mapCurrMasks[pair.first];
			auto pt1 = pCurrIns->pt;
			testCurrID.insert(pair.first);
			GOMAP::GO2D g = pG->Project2D(Kc, Rc, tc);
			g.rect = g.CalcRect();
			auto pt2 = cv::Point(g.rect.x + g.rect.width / 2, g.rect.y + g.rect.height/ 2);
			cv::line(cColorImg, pt1, pt2, cv::Scalar(255, 0, 0), 3);
			cv::rectangle(cColorImg, g.rect, cv::Scalar(255), 2);
			cv::Size newSize(cvRound(g.major * chi), cvRound(g.minor * chi));
			cv::ellipse(cColorImg, pt2, newSize, g.angle_deg, 0, 360, cv::Scalar(255, 255, 0), 3);
			GaussianVisualizer::visualize2D(cColorImg, pG, Kc, Rc, tc, cv::Scalar(255, 0, 255), 1.0, -1);
		}
		//Curr GO

		//시각화
		cv::Mat resImage;
		SLAM->VisualizeMatchingImage(resImage, pColorImg, cColorImg, vecPairVisualizedMatches, mapName, num_vis);
	}

	void AssociationManager::VisualizeErrorAssociation(EdgeSLAM::SLAM* SLAM, BoxFrame* pNewBF, BoxFrame* pPrevBF
		, InstanceMask* pCurrSeg, InstanceMask* pPrevSeg, std::map<std::pair<int, int>, float>& mapErrCase, std::string mapName, int num_vis) 
	{
		auto pPrevKF = pPrevBF->mpRefKF;
		auto pCurrKF = pNewBF->mpRefKF;

		cv::Mat pColorImg = pPrevBF->img.clone();
		cv::Mat cColorImg = pNewBF->img.clone();

		auto mapCurrMasks = pCurrSeg->FrameInstances.Get();
		auto mapPrevMasks = pPrevSeg->FrameInstances.Get();

		std::vector<std::pair<cv::Point2f, cv::Point2f>> vecPairVisualizedMatches;
		for (auto pair : mapErrCase)
		{
			int pid = pair.first.first;
			int cid = pair.first.second;
			float iou = pair.second;

			
			auto pCurrIns = mapCurrMasks[cid];
			auto pt1 = pCurrIns->pt;

			auto pPrevIns = mapPrevMasks[pid];
			auto pt2 = pPrevIns->pt;

			cv::rectangle(cColorImg, pCurrIns->rect, cv::Scalar(255, 255, 0), 2);
			cv::rectangle(pColorImg, pPrevIns->rect, cv::Scalar(255, 255, 0), 2);
			
			vecPairVisualizedMatches.push_back(std::make_pair(pt2, pt1));
		}
		cv::Mat resImage;
		SLAM->VisualizeMatchingImage(resImage, pColorImg, cColorImg, vecPairVisualizedMatches, mapName, num_vis);
	}

	bool AssociationManager::CheckAddNewInstance(std::map<int, FrameInstance*>& mapInstatnces, FrameInstance* pNew) {
		int res = -1;

		std::map<int, float> ious;

		float area = pNew->area;
		cv::Mat sammask = pNew->mask;
		for (auto pair : mapInstatnces)
		{
			auto id = pair.first;
			auto pIns = pair.second;

			cv::Mat cmask = pIns->mask;

			cv::Mat overlap = sammask & cmask;
			float nOverlap = (float)cv::countNonZero(overlap);

			//겹치는게 없으면 무시
			if (nOverlap == 0)
				continue;

			float iou = 0.0;

			if (id == 0)
			{
				iou = nOverlap / area;
			}
			if(id > 0) {
				cv::Mat total = sammask | cmask;
				float nUnion = (float)cv::countNonZero(total);
				iou = nOverlap / nUnion;
			}
			ious[id] = iou;
		}

		bool bres1 = ious[0] > 0.5;
		bool bres2 = true;
		/*if (!bres1)
			std::cout << "bg err " << ious[0] << std::endl;*/
		for (auto pair : ious)
		{
			if (pair.first == 0)
				continue;
			float iou = pair.second;
			if (iou > 0.5)
			{
				//std::cout << iou << std::endl;
				bres2 = false;
				break;
			}
		}

		//백그라운드에 일분에 속하면서
		//기존 마스크와 하나도 안겹쳐야 함.

		return bres1 & bres2;
	}

	int AssociationManager::AddNewInstance(InstanceMask* pFrame, FrameInstance* pNew) {
		int nNewID = pFrame->mnMaxId++;
		pFrame->FrameInstances.Update(nNewID, pNew);
		pFrame->GaussianMaps.Update(nNewID, nullptr);
		//pFrame->MapInstances.Update(nNewID, nullptr);
		//mapInstatnces[nNewID] = pNew;
		return nNewID;
		/*return;
		auto pBG = pFrame->FrameInstances.Get(0);
		cv::Mat maskBG = pBG->mask.clone();
		maskBG -= pNew->mask;*/

		//background update
		//인스턴스에서 전체 마스크 이용하는지 확인 필요
		//pFrame->mask += (pNew->mask / 255) * nNewID;
	}

	void AssociationManager::ConvertContour(const cv::Rect& rect, std::vector<cv::Point>& contour){
		cv::Point2f pt1(rect.x, rect.y);
		cv::Point2f pt2(rect.x + rect.width, rect.y);
		cv::Point2f pt3(rect.x + rect.width, rect.y + rect.height);
		cv::Point2f pt4(rect.x, rect.y + rect.height);
		contour.push_back(pt1);
		contour.push_back(pt2);
		contour.push_back(pt3);
		contour.push_back(pt4);
	}
	bool AssociationManager::CheckOverlap(const cv::Rect& rect, const std::vector<cv::Point>& contour) {
		bool bRes = false;

		for (auto pt : contour)
		{
			if (rect.contains(pt))
			{
				bRes = true;
				break;
			}
		}
		return bRes;
	}
	bool AssociationManager::CheckOverlap(const cv::Rect& rect1, const cv::Rect& rect2){
		std::vector<cv::Point> c2;
		ConvertContour(rect2, c2);

		return CheckOverlap(rect1, c2);
	}
	bool AssociationManager::IsContain(const cv::Rect& rect1, const cv::Rect& rect2) {
		std::vector<cv::Point> c2;
		ConvertContour(rect2, c2);
		bool b = true;
		for (auto pt : c2)
		{
			if (!rect1.contains(pt))
			{
				b = false;
				break;
			}
		}
		return b;
	}

	void AssociationManager::AssociationWithSAM(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key
		, const int id, const int id2, const int _type
		, const std::string& mapName, const std::string& userName, BoxFrame* pNewBF 
		, InstanceMask* pCurrSegMask, std::map<int, FrameInstance*>& mapSamInstances, bool bShow)
	{
		
		auto pCurrSegInstance = pCurrSegMask->FrameInstances.Get();
		
		//맵 또는 프레임 정보 획득
		InstanceType type = (InstanceType)_type;
		std::string stype1 = "reqSAM_RAFT";
		std::string stype2 = "reqSAM_SEG";
		if (type == InstanceType::MAP){
			stype1 = "reqSAM_MAP";
		}

		auto pNewSamMask = new InstanceMask(); //샘 정보를 추가. 아이디는 새로 추가되는 정보로
		InstanceMask* pReqMask = nullptr; //map or raft
		InstanceMask* pReqSegMask = nullptr; //segmentation of previous frame
		std::map<int, FrameInstance*> pReqSAMInstance;
		if (pNewBF->mapMasks.Count(stype1))
		{
			pReqMask = pNewBF->mapMasks.Get(stype1);
			pReqSegMask = pNewBF->mapMasks.Get(stype2);
			pReqSAMInstance = pReqMask->FrameInstances.Get();
		}
		else
			return;
		
		//이전 프레임 정보 획득
		if (!ObjSLAM->MapKeyFrameNBoxFrame.Count(id2)) {
			return;
		}
		auto pPrevBF = ObjSLAM->MapKeyFrameNBoxFrame.Get(id2);
		auto pPrevSegMask = pPrevBF->mapMasks.Get("yoloseg");
		auto pPrevSegInstance = pPrevSegMask->FrameInstances.Get();
		auto pCurrKF = pNewBF->mpRefKF;
		auto pPrevKF = pPrevBF->mpRefKF;
		
		//SAM & FRAME + MAP(reqSAM)
		//std::map<int, std::map<int, AssoMatchRes*>> res; //map<pid, map<cid, asso>>
		std::map<std::pair<int, int>, AssoMatchRes*> res, mapSuccess;
		std::map<int, int> mnMatchedIdx;//sid, pid
		std::set<int> mnAddCurrIdx;
		//std::map<int, std::set<int>> mapMatchSAMnReq; //sam id, set<req id>
		std::map<int, std::set<FrameInstance*>> mapCurrMatchRes; //cid, previnstance
		CalculateIOU(pReqSAMInstance, mapSamInstances, res);
		EvaluateMatchResults(res, mapSuccess);

		//인스턴스 비교용
		//cv::Mat vimg = pNewBF->img.clone();
		//for (auto pair : pCurrSegInstance)
		//{
		//	auto pIns = pair.second;
		//	cv::rectangle(vimg, pIns->rect, cv::Scalar(0, 0, 255), -1);
		//	//cv::putText(vimg, std::to_string(pGO->id), pIns->pt, 2, 1.3, cv::Scalar(0, 0, 255), 2);
		//}
		//for (auto pair : pReqSAMInstance)
		//{
		//	auto pIns = pair.second;
		//	auto rect = pIns->rect;

		//	cv::rectangle(vimg, rect, cv::Scalar(255, 0, 0), 5);
		//	//cv::putText(vimg, std::to_string(pGO->id), pIns->pt, 2, 1.3, cv::Scalar(0, 0, 255), 2);
		//}
		//for (auto pair : mapSamInstances)
		//{
		//	auto pIns = pair.second;
		//	auto rect = pIns->rect;

		//	cv::rectangle(vimg, rect, cv::Scalar(255, 255, 0), 5);
		//}
		//인스턴스 비교용

		//SAM & prev
		
		for (auto pair : mapSuccess)
		{
			auto pid = pair.first.first;
			auto sid = pair.first.second;
			
			auto ares = pair.second;

			auto pReqIns = pReqSAMInstance[pid];
			auto pSAMIns = mapSamInstances[sid];
			if (ares->iou > 0.5)
			{
				ares->res = true;
				//샘 인스턴스가 중복될 수 있음. 확인 필요
				mnMatchedIdx[sid]=pid;
				/*mapMatchSAMnReq[sid].insert(pid);
				cv::rectangle(vimg, pSAMIns->rect, cv::Scalar(0, 255, 0), 5);*/
			}
		}
		
		int tempVisID = 6;
		if (type == InstanceType::SEG)
			tempVisID++;
		
		//test visualization

		//SAM & CURR
		pReqMask->mapResAssociation.clear();
		pReqSegMask->mapResAssociation.clear();
		std::map<int, FrameInstance*> tempPrevInstance = pReqSAMInstance;
		if (type == InstanceType::SEG)
			tempPrevInstance = pPrevSegInstance;
		for (auto pair : mapSamInstances)
		{
			auto sid = pair.first;
			auto pNewSAM = pair.second;

			//???????????????
			if (CheckAddNewInstance(pCurrSegInstance, pNewSAM) && mnMatchedIdx.count(sid))
			{
				int nNewID = AddNewInstance(pCurrSegMask, pNewSAM);
				int pid = mnMatchedIdx[sid];
				pCurrSegInstance[nNewID] = pNewSAM;

				auto ares = mapSuccess[std::make_pair(pid, sid)];
				ares->id2 = nNewID;
				//어소시에이션 결과 교체
				pReqMask->mapResAssociation[pid] = ares;
				pReqSegMask->mapResAssociation[pid] = ares;
				/*pReqMask->mapResAssociation[pid]->id2 = nNewID;
				pReqSegMask->mapResAssociation[pid]->id2 = nNewID;*/

				pNewSamMask->FrameInstances.Update(nNewID, pNewSAM);
				pNewSamMask->GaussianMaps.Update(nNewID, nullptr);

			}
		}
		//background update

		//인스턴스 화면 시각화
		//SLAM->VisualizeImage(mapName, vimg, tempVisID);

		//Global Map Update
		if (type == InstanceType::SEG)
		{
			UpdateGaussianObjectMap(pReqSegMask, pNewSamMask, type);

			for (auto pair : pReqSegMask->mapResAssociation)
			{
				auto ares = pair.second;
				auto pid = pair.first;
				auto cid = ares->id2;

				auto pG = pReqSegMask->GaussianMaps.Get(pid);
				if (!pG)
					continue;
				auto pPrevG = pPrevSegMask->GaussianMaps.Get(pid);
				auto pCurrG = pCurrSegMask->GaussianMaps.Get(cid);
				if (pG)
					pPrevSegMask->GaussianMaps.Update(pid, pG);
				if(!pPrevG)
					pPrevSegMask->GaussianMaps.Update(pid, pG);
				if (!pCurrG)
					pCurrSegMask->GaussianMaps.Update(cid, pG);
			}

		}
		if (type == InstanceType::MAP)
		{
			UpdateGaussianObjectMap(pReqSegMask, pCurrSegMask, type);
		}
		
		//Visualization
		if(type == InstanceType::SEG)
			VisualizeAssociation(SLAM, pNewBF, pPrevBF, pCurrSegMask, mapCurrMatchRes, mapName, 2);
	}
	void AssociationManager::Association(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key, const int id, const int id2
		, const std::string& mapName, const std::string& userName, BoxFrame* pNewBF, BoxFrame* pPrevBF, InstanceMask* pPrevSegMask
		, InstanceMask* pCurrSegMask, InstanceMask* pRaft, bool bShow){
		
		//std::map<GOMAP::GaussianObject*, FrameInstance*> mapInstance;
		//ProjectObjectMap(pCurrSegMask, pPrevSegMask, pNewBF, pPrevBF, mapInstance);

		//iou matching

		//raft
		auto pPrevGO = pPrevSegMask->GaussianMaps.Get();
		auto pCurrSegInstance = pCurrSegMask->FrameInstances.Get();
		auto pPrevSegInstance = pPrevSegMask->FrameInstances.Get();

		//가우시안 맵. pCurrGO는 MAP을 이용한 연결에서 연결된 경우 프레임끼리 매칭 될 때 이전 프레임에 맵을 전달하는 용도
		auto pCurrGOs = pCurrSegMask->GaussianMaps.Get();
		auto pPrevGOs = pPrevSegMask->GaussianMaps.Get();

		auto pCurrKF = pNewBF->mpRefKF;
		auto pPrevKF = pPrevBF->mpRefKF;
		std::map<int, FrameInstance*> mapRaftInstance;
		const cv::Mat flow = pRaft->mask;

		ConvertMaskWithRAFT(pPrevSegInstance, mapRaftInstance, pPrevKF, flow);
		//prev의 GO, RAFT check

		//iou matching
		std::map<std::pair<int, int>, AssoMatchRes*> res, mapSuccess;
		CalculateIOU(mapRaftInstance, pCurrSegInstance, res);
		EvaluateMatchResults(res, mapSuccess);

		////Error case test 중
		////count
		//std::map<int, int> mapCountPrev, mapCountCurr;
		//std::map<std::pair<int, int>, float> mapErrCase;
		//for (auto pair : res)
		//{
		//	auto pid = pair.first;
		//	
		//	for (auto pair2 : pair.second)
		//	{
		//		auto cid = pair2.first;
		//		auto ares = pair2.second;
		//		if(ares->iou > 0.5){
		//			mapCountCurr[cid]++;
		//			mapCountPrev[pid]++;
		//		}
		//	}
		//}
		//for (auto pair : res)
		//{
		//	auto pid = pair.first;
		//	if (pid == 0)
		//		continue;
		//	for (auto pair2 : pair.second)
		//	{
		//		auto cid = pair2.first;
		//		if (cid == 0)
		//			continue;
		//		int nCount = mapCountCurr[cid];
		//		auto ares = pair2.second;
		//		if (nCount > 1 && ares->iou > 0.5)
		//		{
		//			auto keypair = std::make_pair(pid, cid);
		//			mapErrCase[keypair] = ares->iou;
		//			//std::cout <<keypair.first<<" "<<keypair.second << " == " << ares->iou <<" "<<(int)pPrevSegInstance[pid]->type << std::endl;
		//		}
		//	}
		//}
		//Error case test 중
		 
		//iou > th
		std::map<int, std::set<FrameInstance*>> mapCurrMatchRes; //cid, previnstance
		std::map<FrameInstance*, std::set<int>> mapMapMatchRes;

		//성공한 것은 뉴마스크에(original mask), 샘으로 추가 비교가 필요한 것은 샘마스크에 기록(라프트 인스턴스가 기록)
		//프레브가 여러 컬과 중복되는 경우는? 일단 확인해보고 처리하기~~~~~~~~~~~~~~~~~~~~~~ 확인 후 삭제
		//컬의 중복이 거의 같은 프레브가 존재하는 경우. 샘으로 잘못 추가된 경우가 대부분일 듯? 샘 확인 후 삭제
		auto pNewMask = new InstanceMask();
		auto pSAMRaftMask = new InstanceMask();
		auto pSAMSegMask = new InstanceMask();
		
		std::set<int> sAlready;
		for (auto pair : mapSuccess)
		{
			auto pid = pair.first.first;
			auto cid = pair.first.second;

			if (!sAlready.count(pid))
				sAlready.insert(pid);

			auto pPrevIns = pPrevSegInstance[pid];
			auto pRaftIns = mapRaftInstance[pid];
			auto pPrevGO = pPrevGOs[pid];

			auto ares = pair.second;

			if (ares->iou > 0.5)
			{
				if (cid == 0)
				{
					ares->req = true;
					ares->res = false;
					pSAMRaftMask->FrameInstances.Update(pid, pRaftIns);
					pSAMRaftMask->GaussianMaps.Update(pid, pPrevGO);
					pSAMRaftMask->mapResAssociation[pid] = ares;

					pSAMSegMask->FrameInstances.Update(pid, pPrevIns);
					pSAMSegMask->GaussianMaps.Update(pid, pPrevGO);
					pSAMSegMask->mapResAssociation[pid] = ares;
				}
				else {
					ares->res = true;
					ares->req = false;
					pNewMask->FrameInstances.Update(pid, pPrevIns);
					pNewMask->GaussianMaps.Update(pid, pPrevGO);
					pNewMask->mapResAssociation[pid] = ares;
				}
				mapMapMatchRes[pPrevIns].insert(id);
				mapCurrMatchRes[cid].insert(pPrevIns);
			}
		}
		
		//샘마스크에서 raft가 아니라 오리지널을 접근 가능해야 함.

		////test code
		//for (auto pair : mapMapMatchRes)
		//{
		//	int n = pair.second.size();
		//	if (n > 1){
		//		auto pid = pair.first;
		//		std::cout << "frame = " << pid <<" - " << n << " ==";
		//		for (auto id : pair.second)
		//		{
		//			std::cout << id << ",";
		//		}
		//		std::cout<<std::endl;
		//	}
		//}
		//for (auto pair : mapCurrMatchRes)
		//{
		//	int cid = pair.first;
		//	if (cid == 0)
		//		continue;
		//	int n = pair.second.size();
		//	if (n > 1)
		//		std::cout <<"match = "<<cid<<" - " << n << std::endl;
		//}
		////test code

		////assocation result
		pNewBF->mapMasks.Update("Frame", pNewMask);
		pNewBF->mapMasks.Update("reqSAM_RAFT", pSAMRaftMask);
		pNewBF->mapMasks.Update("reqSAM_SEG" , pSAMSegMask);

		//request SAM
		cv::Mat ptdata(0, 1, CV_32FC1);
		auto pSAMIns = pSAMRaftMask->FrameInstances.Get();
		for (auto pair : pSAMIns)
		{
			auto p = pair.second;
			auto rect = p->rect;
			cv::Mat temp = cv::Mat::zeros(4, 1, CV_32FC1);
			temp.at<float>(0) = rect.x;
			temp.at<float>(1) = rect.y;
			temp.at<float>(2) = rect.x + rect.width;
			temp.at<float>(3) = rect.y + rect.height;
			ptdata.push_back(temp);
		}
		
		if (ptdata.rows > 0) {
			//reqest
			int nobj = ptdata.rows;
			ptdata.push_back(cv::Mat::zeros(1500 - nobj, 1, CV_32FC1));
			//id2 : prev frame, type : (1) frame, (2) map
			std::string tsrc = userName + ".Image." + std::to_string(nobj)+"."+std::to_string(id2)+"."+std::to_string((int)InstanceType::SEG);
			auto sam2key = "reqsam2";
			std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();
			auto du_upload = Utils::SendData(sam2key, tsrc, ptdata, id, 15, t_start.time_since_epoch().count());
		}

		//check sam request

		//update gaussian object
		//기존 마스크에 추가된 내용에 대해서만 갱신
		//prev와 curr이어야 함. raft는 안됨.
		UpdateGaussianObjectMap(pNewMask, pCurrSegMask, InstanceType::SEG);
		for (auto pair : pNewMask->mapResAssociation)
		{
			auto ares = pair.second;
			auto pid = pair.first;
			auto cid = ares->id2;

			auto pG = pNewMask->GaussianMaps.Get(pid);
			if (pG)
				pPrevSegMask->GaussianMaps.Update(pid, pG);
		}
		//newmask정보를 추가하기

		AssociationWithPrev(SLAM, ObjSLAM, key, id, mapName, userName, pNewBF, pPrevBF, pPrevSegMask, pCurrSegMask, sAlready);
		//new mask에 객체 추가.
		
		//매칭을 통한 보정

		//cv::Mat vimg = pNewBF->img.clone();
		//VisualizeInstance(mapRaftInstance, pCurrSegInstance, vimg);
		//SLAM->VisualizeImage(mapName, vimg, 0);

		//visualization
		//VisualizeAssociation(SLAM, pNewBF, pPrevBF, pCurrSegMask, mapCurrMatchRes, mapName);
		//VisualizeErrorAssociation(SLAM, pNewBF, pPrevBF, pCurrSegMask, pPrevSegMask, mapErrCase, mapName, 6);

		//std::cout << "Association::test = " <<id<<"=="<< nGO << " == " << mapRaftInstance.size() << " == " << mapMapMatchRes.size() << " " << " ||" << pCurrSegInstance.size() << std::endl;
	}
	void AssociationManager::AssociationWithPrev(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key, const int id
		, const std::string mapName, const std::string userName, BoxFrame* pNewBF, BoxFrame* pPrevBF
		, InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, const std::set<int>& sAlready)
	{
		auto mapCurrSegInstance = pCurrSegMask->FrameInstances.Get();
		auto mapPrevSegInstance = pPrevSegMask->FrameInstances.Get();
		auto tempGOs = pPrevSegMask->GaussianMaps.Get();

		std::map<int, GOMAP::GaussianObject*> mapPrevGOs;
		for (auto pair : tempGOs)
		{
			if (sAlready.count(pair.first))
				continue;
			auto pG = pair.second;
			if (pG)
			{
				mapPrevGOs[pair.first] = pG;
			}
		}
		std::map<int, GOMAP::GO2D> mapPrevGO2Ds, mapCurrGO2Ds;
		std::map<int, FrameInstance*> mapPrevMapInstance, mapCurrMapInstance;

		//keyframe
		auto pPrevKF = pPrevBF->mpRefKF;
		auto pCurrKF = pNewBF->mpRefKF;

		//convert prev frame
		GetObjectMap2Ds(mapPrevGOs, pPrevBF, mapPrevGO2Ds);
		ConvertFrameInstances(mapPrevGO2Ds, pPrevBF, mapPrevMapInstance);
		//convert curr frame
		GetObjectMap2Ds(mapPrevGOs, pNewBF, mapCurrGO2Ds);
		ConvertFrameInstances(mapCurrGO2Ds, pNewBF, mapCurrMapInstance);

		//매칭 정보
		cv::Mat cimg = pNewBF->img.clone();
		cv::Mat pimg = pPrevBF->img.clone();
		std::vector<std::pair<cv::Point2f, cv::Point2f>> vecPairVisualizedMatches;
		std::map<int, std::vector<cv::Point2f>> mapPrevKPs, mapCurrKPs;
		std::map<int, cv::Mat> mapPrevDescs, mapCurrDescs;

		//포인트 매칭
		//이전 프레임 : 인스턴스 내에서
		//현재 프레임 : 현재 프레임의 가능 영역 내에서 
		float chi = sqrt(5.991);
		for (auto pair : mapPrevGOs)
		{
			auto pid = pair.first;
			auto pG = pair.second;
			auto pPrevSeg = mapPrevSegInstance[pid];
			auto pCurrMap2D = mapCurrGO2Ds[pG->id];
			
			pCurrMap2D.rect = pCurrMap2D.CalcRect(chi);
			auto crect = pCurrMap2D.rect;
			auto prect = pPrevSeg->rect;

			cv::rectangle(cimg, crect, cv::Scalar(255, 0, 0), 2);
			cv::rectangle(pimg, prect, cv::Scalar(255, 0, 0), 2);

			cv::Mat pdesc = cv::Mat::zeros(0, 32, CV_8UC1);
			cv::Mat cdesc = cv::Mat::zeros(0, 32, CV_8UC1);
			std::vector<cv::Point2f> vecPrev, vecCurr;

			for (int i = 0; i < pPrevKF->mvKeys.size(); i++) {
				auto kp = pPrevKF->mvKeys[i];
				if (prect.contains(kp.pt))
				{
					vecPrev.push_back(kp.pt);
					pdesc.push_back(pPrevKF->mDescriptors.row(i));
					cv::circle(pimg, kp.pt, 3, cv::Scalar(0, 0, 255), -1);
				}
			}
			for (int i = 0; i < pCurrKF->mvKeys.size(); i++) {
				auto kp = pCurrKF->mvKeys[i];
				if (crect.contains(kp.pt))
				{
					vecCurr.push_back(kp.pt);
					cdesc.push_back(pCurrKF->mDescriptors.row(i));
					cv::circle(cimg, kp.pt, 3, cv::Scalar(0, 0, 255), -1);
				}
			}
			//매칭
			std::vector<std::pair<int, int>> vecMatches;
			ObjectMatcher::SearchInstance(pdesc, cdesc, vecMatches);

			std::vector<cv::Point2f> vecMatchCurrPoints;

			for (int i = 0; i < vecMatches.size(); i += 1)
			{
				auto pair2 = vecMatches[i];
				auto pt1 = vecPrev[pair2.first];
				auto pt2 = vecCurr[pair2.second];
				vecMatchCurrPoints.push_back(pt2);
				vecPairVisualizedMatches.push_back(std::make_pair(pt1, pt2));
			}
			auto cRect = cv::boundingRect(vecMatchCurrPoints);
			cv::rectangle(cimg, cRect, cv::Scalar(255, 255, 0), 3);
			/*for (auto pair2 : vecMatches)
			{
				auto pt1 = vecPrev[pair2.first];
				auto pt2 = vecCurr[pair2.second];
				vecPairVisualizedMatches.push_back(std::make_pair(pt1, pt2));
			}*/

			mapPrevKPs[pid] = vecPrev;
			mapCurrKPs[pid] = vecCurr;
			mapPrevDescs[pid] = pdesc;
			mapCurrDescs[pid] = cdesc;
			
		}

		//시각화
		cv::Mat resImage;
		SLAM->VisualizeMatchingImage(resImage, pimg, cimg, vecPairVisualizedMatches, mapName, 0);

		////맵을 현재 프레임에 프로젝션한 것, 현재 프레임의 인스턴스
		//std::map<int, std::map<int, AssoMatchRes*>> res; //pid, map<cid, iou>
		//CalculateIOU(mapCurrMapInstance, mapCurrSegInstance, res);

		////매칭 연결.
		////다대다 매칭 문제 해결 필요

		////업데이트

		//////시각화
		//////test visualization
		////cv::Mat vimg = pNewBF->img.clone();
		////for (auto pair : mapCurrMapInstance)
		////{
		////	auto pGO = mapPrevGOs[pair.first];
		////	auto pIns = pair.second;
		////	cv::rectangle(vimg, pIns->rect, cv::Scalar(255, 0, 0), 2);
		////	cv::putText(vimg, std::to_string(pGO->id), pIns->pt, 2, 1.3, cv::Scalar(0, 0, 255), 2);
		////}
		////for (auto pair : mapCurrSegInstance)
		////{
		////	auto pIns = pair.second;
		////	cv::rectangle(vimg, pIns->rect, cv::Scalar(255, 255, 0), 2);
		////}
		////SLAM->VisualizeImage(mapName, vimg, 5);
		////test visualization

		////여기서 실패한 객체에 대해서 다음 로컬 맵에서 매칭하도록 해야 함.
	}
	void AssociationManager::AssociationWithMAP(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key, const int id
		, const std::string mapName, const std::string userName, BoxFrame* pNewBF, BoxFrame* pPrevBF
		, InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, bool bShow) {

		//prev 정보를 내부에서 얻도록 변경.
		//auto pCurrKF = pNewBF->mpRefKF;
		//auto pPrevKF = pPrevBF->mpRefKF;
		//auto pCurrSegInstance = pCurrSegMask->FrameInstances.Get();

		//std::map<int, GOMAP::GaussianObject*> mapLocalGOs;
		//std::map<int, GOMAP::GO2D> mapLocalPrevGO2Ds, mapLocalCurrGO2Ds;
		//std::map<int, FrameInstance*> mapMapPrevFrameInstance, mapMapCurrFrameInstance;
		//GetLocalObjectMaps(pPrevSegMask, mapLocalGOs);
		////convert prev frame
		//GetObjectMap2Ds(mapLocalGOs, pPrevBF, mapLocalPrevGO2Ds);
		//ConvertFrameInstances(mapLocalPrevGO2Ds, pPrevBF, mapMapPrevFrameInstance);
		////convert curr frame
		//GetObjectMap2Ds(mapLocalGOs, pNewBF, mapLocalCurrGO2Ds);
		//ConvertFrameInstances(mapLocalCurrGO2Ds, pNewBF, mapMapCurrFrameInstance);

		////매칭 정보
		//std::map<int, std::vector<cv::Point2f>> mapPrevKPs, mapCurrKPs;
		//std::map<int, cv::Mat> mapPrevDescs, mapCurrDescs;

		////test visualization
		//float chi = sqrt(5.991);
		//cv::Mat vimg = pNewBF->img.clone();
		//

		//std::vector<cv::Point2f> mvPrevKeys, mvCurrKeys;
		//cv::Mat prevDesc = cv::Mat::zeros(0, 32, CV_8UC1);
		//cv::Mat currDesc = cv::Mat::zeros(0, 32, CV_8UC1);

		//for (auto pair : mapMapCurrFrameInstance)
		//{
		//	auto pGO = mapLocalGOs[pair.first];
		//	auto pIns = pair.second;
		//	cv::rectangle(vimg, pIns->rect, cv::Scalar(255, 0, 0), 2);

		//	auto g = mapLocalPrevGO2Ds[pair.first];
		//	cv::Size newSize(cvRound(g.major * chi), cvRound(g.minor * chi));
		//	cv::ellipse(vimg, pIns->pt, newSize, g.angle_deg, 0, 360, cv::Scalar(255, 255, 0), 3);
		//	cv::putText(vimg, std::to_string(pGO->id), pIns->pt, 2, 1.3, cv::Scalar(0, 0, 255), 2);

		//	//현재 프레임의 포인트 후보군 획득

		//	//이전 프레임에서 

		//	////피쳐 출력
		//	//g.GetRect(chi);
		//	//auto rect = g.rect;
		//	//cv::ellipse(vimg2, pIns->pt, newSize, g.angle_deg, 0, 360, cv::Scalar(255, 255, 0), 3);
		//	//for(int i = 0; i < pKF->mvKeys.size(); i++)
		//	//{
		//	//	auto kp = pKF->mvKeys[i];
		//	//	if (rect.contains(kp.pt))
		//	//	{
		//	//		mvPrevKeys.push_back(kp.pt);
		//	//		prevDesc.push_back(pKF->mDescriptors.row(i));
		//	//		cv::circle(vimg2, kp.pt, 5, cv::Scalar(255, 0, 0), -1);
		//	//	}
		//	//}
		//}
		//for (auto pair : pCurrSegInstance)
		//{
		//	auto pIns = pair.second;
		//	cv::rectangle(vimg, pIns->rect, cv::Scalar(255, 255, 0), 2);

		//	/*cv::rectangle(vimg2, pIns->rect, cv::Scalar(0, 0, 255), 2);
		//	for (int i = 0; i < pKF->mvKeys.size(); i++)
		//	{
		//		auto kp = pKF->mvKeys[i];
		//		if (pIns->rect.contains(kp.pt))
		//		{
		//			mvCurrKeys.push_back(kp.pt);
		//			currDesc.push_back(pKF->mDescriptors.row(i));
		//			cv::circle(vimg2, kp.pt, 3, cv::Scalar(0, 0, 255), -1);
		//		}
		//	}*/
		//}

		////전체 매칭 테스트

		////전체 매칭 테스트

		//SLAM->VisualizeImage(mapName, vimg, 5);
		////SLAM->VisualizeImage(mapName, vimg2, 1);
		////test visualization

		//////수정 버전
		////std::map<GOMAP::GaussianObject*, FrameInstance*> mapInstance; //type은 MAP
		////ProjectObjectMap(pCurrSegMask, pPrevSegMask, pNewBF, mapInstance);

		////std::map<int, GOMAP::GaussianObject*> mapMap3D;
		////std::map<int, FrameInstance*> mapMap2D;
		////for (auto pair : mapInstance)
		////{
		////	int id = pair.first->id;
		////	mapMap2D[id] = pair.second;
		////	mapMap3D[id] = pair.first;
		////}
		//////수정 버전

		////MAP MASK 생성
		////id는 GO의 global id임.
		//auto pMapMask = new InstanceMask();
		//auto pSamMask = new InstanceMask();

		////iou matching
		//std::map<int, std::map<int, AssoMatchRes*>> res; //pid, map<cid, iou>
		//CalculateIOU(mapMapCurrFrameInstance, pCurrSegInstance, res);

		////iou > th
		////시각화 코드는 수정이 필요할 듯
		//std::map<std::pair<int, int>, GOMAP::GaussianObject*> mapUpdatedGO; //pair<pid, cid>
		//std::map<int, std::set<FrameInstance*>> mapCurrMatchRes;
		////std::map<FrameInstance*, std::set<int>> mapMapMatchRes;
		//for (auto pair : res)
		//{
		//	auto mid = pair.first;
		//	auto pPrevIns = mapMapPrevFrameInstance[mid];
		//	auto pCurrIns = mapMapCurrFrameInstance[mid];
		//	auto pGO = mapLocalGOs[mid];

		//	//중복 문제에 대한 해결이 필요함.
		//	for (auto pair2 : pair.second)
		//	{
		//		auto cid = pair2.first;
		//		auto ares = pair2.second;

		//		if (ares->iou > 0.5)
		//		{

		//			if (cid == 0)
		//			{
		//				//req SAM
		//				ares->req = true;
		//				ares->res = false;

		//				pSamMask->FrameInstances.Update(mid, pCurrIns);
		//				pSamMask->GaussianMaps.Update(mid, pGO);
		//			}
		//			else {
		//				//matching
		//				ares->res = true;
		//				ares->req = false;

		//				pMapMask->FrameInstances.Update(mid, pPrevIns);
		//				pMapMask->GaussianMaps.Update(mid, pGO);

		//				mapUpdatedGO[std::make_pair(mid, cid)] = nullptr;
		//			}

		//			//mapMapMatchRes[pF].insert(cid);
		//			mapCurrMatchRes[cid].insert(pPrevIns);

		//		}
		//	}
		//}

		//pNewBF->mapMasks.Update("ObjMap", pMapMask);
		//pNewBF->mapMasks.Update("reqSAM_MAP", pSamMask);

		////request SAM
		//cv::Mat ptdata(0, 1, CV_32FC1);
		//auto pSAMIns = pSamMask->FrameInstances.Get();
		//for (auto pair : pSAMIns)
		//{
		//	auto p = pair.second;
		//	auto rect = p->rect;
		//	cv::Mat temp = cv::Mat::zeros(4, 1, CV_32FC1);
		//	temp.at<float>(0) = rect.x;
		//	temp.at<float>(1) = rect.y;
		//	temp.at<float>(2) = rect.x + rect.width;
		//	temp.at<float>(3) = rect.y + rect.height;
		//	ptdata.push_back(temp);
		//}

		//if (ptdata.rows > 0) {
		//	//reqest
		//	int nobj = ptdata.rows;
		//	ptdata.push_back(cv::Mat::zeros(1500 - nobj, 1, CV_32FC1));
		//	//id2 : prev frame, type : (1) frame, (2) map
		//	
		//	std::string tsrc = userName + ".Image." + std::to_string(nobj) + "." + std::to_string(pPrevBF->mnId) + "." + std::to_string((int)InstanceType::MAP);
		//	auto sam2key = "reqsam2";
		//	std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();
		//	auto du_upload = Utils::SendData(sam2key, tsrc, ptdata, id, 15, t_start.time_since_epoch().count());
		//}

		//UpdateGaussianObjectMap(mapUpdatedGO, pMapMask, pCurrSegMask, InstanceType::MAP);
		////VisualizeAssociation(SLAM, pNewBF, pPrevBF, pCurrSegMask, mapCurrMatchRes, mapName, 6);
		//return;
		//

		//

		////update
		////std::cout << "AssociationWithMAP::test = " << mapMap2D.size() << " " << mapMapMatchRes.size() << std::endl;

	}
	
	void AssociationManager::ExtractRegionFeatures(EdgeSLAM::KeyFrame* pKF, const cv::Rect& rect, std::vector<cv::KeyPoint>& vecKPs, cv::Mat& desc) {
		desc = cv::Mat::zeros(0, 32, CV_8UC1);

		for (int i = 0; i < pKF->mvKeys.size(); i++) {
			auto kp = pKF->mvKeys[i];
			if (rect.contains(kp.pt))
			{
				vecKPs.push_back(kp);
				desc.push_back(pKF->mDescriptors.row(i));
			}
		}
	}
	void AssociationManager::ExtractRegionFeatures(EdgeSLAM::KeyFrame* pKF, const cv::Rect& rect
		, std::vector<cv::KeyPoint>& vecKPs, std::vector<EdgeSLAM::MapPoint*>& vecMPs, cv::Mat& desc) {
		desc = cv::Mat::zeros(0, 32, CV_8UC1);

		for (int i = 0; i < pKF->mvKeys.size(); i++) {
			auto kp = pKF->mvKeys[i];
			if (rect.contains(kp.pt))
			{
				vecKPs.push_back(kp);
				desc.push_back(pKF->mDescriptors.row(i));

				auto pMPi = pKF->mvpMapPoints.get(i);
				vecMPs.push_back(pMPi);
				/*if (pMPi && !pMPi->isBad())
				{
					vecMPs.push_back(pMPi);
				}
				else
					vecMPs.push_back(nullptr);*/
			}
		}
	}
	void AssociationManager::ExtractRegionFeatures(EdgeSLAM::KeyFrame* pKF, const cv::RotatedRect& rect, std::vector<cv::KeyPoint>& vecKPs, cv::Mat& desc) {
		desc = cv::Mat::zeros(0, 32, CV_8UC1);

		for (int i = 0; i < pKF->mvKeys.size(); i++) {
			auto kp = pKF->mvKeys[i];
			if (isPointInRotatedRect(kp.pt, rect))
			{
				vecKPs.push_back(kp);
				desc.push_back(pKF->mDescriptors.row(i));
			}
		}
	}
	void AssociationManager::ExtractRegionFeatures(EdgeSLAM::KeyFrame* pKF, const cv::RotatedRect& rect, 
		std::vector<cv::KeyPoint>& vecKPs, std::vector<EdgeSLAM::MapPoint*>& vecMPs, cv::Mat& desc) {
		desc = cv::Mat::zeros(0, 32, CV_8UC1);

		for (int i = 0; i < pKF->mvKeys.size(); i++) {
			auto kp = pKF->mvKeys[i];
			if (isPointInRotatedRect(kp.pt, rect))
			{
				vecKPs.push_back(kp);
				desc.push_back(pKF->mDescriptors.row(i));

				auto pMPi = pKF->mvpMapPoints.get(i);
				if (pMPi && !pMPi->isBad())
				{
					vecMPs.push_back(pMPi);
				}
				else
					vecMPs.push_back(nullptr);
			}
		}
	}
}
