#include <AssociationManager.h>

//EdgeSLAM
#include <Utils.h>
#include <Utils_Geometry.h>
#include <SLAM.h>
#include <Camera.h>
#include <Frame.h>
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

//����þ� ��ü ��
#include <Gaussian/GaussianObject.h>
#include <Gaussian/GaussianMapManager.h>
#include <Gaussian/Visualizer.h>
#include <Gaussian/Optimization/ObjectOptimizer.h>

//evaluation
#include <Eval/EvalObj.h>
#include <vis/VisObject.h>

namespace ObjectSLAM {
	
	ConcurrentMap<int, int> AssociationManager::DebugAssoSeg, AssociationManager::DebugAssoSAM;
	bool AssociationManager::mbSAM = false;
	bool AssociationManager::mbObjBaselineTest = false;

	float sign(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3) {
		return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
	}
	bool pointInTriangle(const cv::Point2f& pt, const cv::Point2f& v1,
		const cv::Point2f& v2, const cv::Point2f& v3) {
		// ������ �̿��� ���
		float d1 = sign(pt, v1, v2);
		float d2 = sign(pt, v2, v3);
		float d3 = sign(pt, v3, v1);

		bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
		bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);

		// ��ȣ�� ��� ������ �ﰢ�� ���ο� ����
		return !(has_neg && has_pos);
	}
	bool isPointInRotatedRect(const cv::Point2f& point, const cv::RotatedRect& rect) {
		// ȸ���� �簢���� ������ 4�� ���
		cv::Point2f vertices[4];
		rect.points(vertices);

		// ���� 4���� �ﰢ�� �ȿ� �ִ��� Ȯ��
		// �簢���� �� ���� �ﰢ������ ������ Ȯ��
		return (pointInTriangle(point, vertices[0], vertices[1], vertices[2]) ||
			pointInTriangle(point, vertices[0], vertices[2], vertices[3]));
	}

	void AssociationManager::AssociationWithSeg(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM,
		const std::string& key, const std::string& mapName, const std::string& userName, AssoFramePairData* pPairData)
	{

		std::cout << "SEG = " << pPairData->toid << std::endl;

		auto pPrevBF = pPairData->mpFrom;
		auto pNewBF = pPairData->mpTo;

		auto pCurrKF = pNewBF->mpRefKF;
		auto pPrevKF = pPrevBF->mpRefKF;

		auto pCurrSegMask = pNewBF->mapMasks.Get("yoloseg");
		auto pPrevSegMask = pPrevBF->mapMasks.Get("yoloseg");

		auto pCurrSegInstance = pCurrSegMask->FrameInstances.Get();
		auto pPrevSegInstance = pPrevSegMask->FrameInstances.Get();

		//����þ� ��. pCurrGO�� MAP�� �̿��� ���ῡ�� ����� ��� �����ӳ��� ��Ī �� �� ���� �����ӿ� ���� �����ϴ� �뵵
		auto pCurrGOs = pCurrSegMask->GaussianMaps.Get();
		auto pPrevGOs = pPrevSegMask->GaussianMaps.Get();

		auto pRaftMask = pPairData->mpRaftIns;

		std::map<int, FrameInstance*> mapRaftInstance;
		const cv::Mat flow = pRaftMask->mask;

		//int didx = 0;
		//DebugAssoSeg.Update(didx, DebugAssoSeg.Get(didx) + 1);
		
		ConvertMaskWithRAFT(pPrevSegInstance, mapRaftInstance, pPrevKF, flow);
		for (auto pair : mapRaftInstance)
		{
			auto pid = pair.first;
			auto pRaftIns = pair.second;
			pRaftMask->FrameInstances.Update(pid, pRaftIns);
			pRaftMask->MapInstances.Update(pid, nullptr);
			pRaftMask->GaussianMaps.Update(pid, nullptr);
		}
		//prev�� GO, RAFT check

		//didx = 1;
		//DebugAssoSeg.Update(didx, DebugAssoSeg.Get(didx) + 1);
		

		//iou matching
		std::map<std::pair<int, int>, AssoMatchRes*> res, mapSuccess;
		CalculateIOU(mapRaftInstance, pCurrSegInstance, res);
		EvaluateMatchResults(res, mapSuccess);

		//didx = 2;
		//DebugAssoSeg.Update(didx, DebugAssoSeg.Get(didx) + 1);
		
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
				AssociationResultType restype;
				if (cid == 0)
				{
					ares->req = true;
					ares->res = false;
					
					pPairData->mapReqRaft[pid] = 0;
					restype = AssociationResultType::RECOVERY;
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

					restype = AssociationResultType::SUCCESS;
					pPairData->mapRaftSeg[pid] = cid;
					/*pNewMask->FrameInstances.Update(pid, pPrevIns);
					pNewMask->GaussianMaps.Update(pid, pPrevGO);
					pNewMask->mapResAssociation[pid] = ares;*/
				}
				pPairData->mapRaftResult[pid] = restype;
				/*mapMapMatchRes[pPrevIns].insert(id);
				mapCurrMatchRes[cid].insert(pPrevIns);*/
			}
		}
		for (auto pair : pPrevSegInstance)
		{
			auto pid = pair.first;
			if (!pPairData->mapRaftResult.count(pid)) {
				pPairData->mapRaftResult[pid] = AssociationResultType::FAIL;
			}
		}

		//didx = 3;
		//DebugAssoSeg.Update(didx, DebugAssoSeg.Get(didx) + 1);

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

		//didx = 4;
		//DebugAssoSeg.Update(didx, DebugAssoSeg.Get(didx) + 1);

		//check sam request
		AssociationPrevMapWithUncertainty(SLAM, pPairData);
		//��Ī ��� ����
		{
			cv::Mat cimg = pNewBF->img.clone();
			cv::Mat pimg = pPrevBF->img.clone();

			std::vector<std::pair<cv::Point2f, cv::Point2f>> vecMatch, vecMatch2;
			auto mapIns = pPairData->mpPrevMapIns->FrameInstances.Get();

			for (auto pair : pPrevSegInstance)
			{
				auto pid = pair.first;
				auto res = pPairData->mapPrevMapResult[pid];

				auto pins = pPrevSegInstance[pid];

				if (res == AssociationResultType::FAIL)
				{
					cv::rectangle(pimg, pins->rect, cv::Scalar(0, 0, 255), 2);
				}
				else {
					auto mins = mapIns[pid];
					auto mpair = std::make_pair(pins->pt, mins->pt);

					if (res == AssociationResultType::SUCCESS)
					{
						cv::rectangle(pimg, pins->rect, cv::Scalar(0, 255, 0), 2);
						cv::rectangle(cimg, mins->rect, cv::Scalar(0, 255, 0), 2);
						vecMatch.push_back(mpair);
					}
					if (res == AssociationResultType::RECOVERY)
					{
						cv::rectangle(pimg, pins->rect, cv::Scalar(0, 255, 255), 2);
						cv::rectangle(cimg, mins->rect, cv::Scalar(0, 255, 255), 2);
						vecMatch2.push_back(mpair);
					}

				}
			}
			
			cv::Mat resImage;
			SLAM->VisualizeMatchingImage(resImage, pimg, cimg, vecMatch, mapName, 0, cv::Scalar(0, 255, 0));
			SLAM->VisualizeMatchingImage(resImage, vecMatch2, mapName, 0, cv::Scalar(0, 255, 255));

			std::stringstream ss;
			ss.str("");
			ss << "../res/asso/" << pPairData->toid << "_" << pPairData->fromid << "_" << 2 << ".png";
			cv::imwrite(ss.str(), resImage);
		}
		//��Ī ��� ����

		//didx = 5;
		//DebugAssoSeg.Update(didx, DebugAssoSeg.Get(didx) + 1);

		std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();
		
		//SAM ���� �� ��Ī �����͸� RaftSeg�� ����ϱ�
		for (auto pair : pPrevSegInstance)
		{
			auto pid = pair.first;
			auto res1 = pPairData->mapRaftResult[pid];
			auto res2 = pPairData->mapPrevMapResult[pid];
			if (res1 == AssociationResultType::FAIL && res2 == AssociationResultType::SUCCESS)
			{
				auto cid = pPairData->mapMapSeg[pid];
				pPairData->mapRaftSeg[pid] = cid;
				pPairData->mapRaftResult[pid] = AssociationResultType::SUCCESS;
			}
			
		}

		//RAFT ��Ī�� prev Map ��Ī �� ���� cid�� fail�� ����
		std::set<int> tempIDs;
		for (auto pair : pPrevSegInstance)
		{
			auto pid = pair.first;
			auto res1 = pPairData->mapRaftResult[pid];
			if (res1 == AssociationResultType::SUCCESS) {
				tempIDs.insert(pPairData->mapMapSeg[pid]);
			}
			/*if(res1 == AssociationResultType::RECOVERY)
			{
				tempIDs.insert(pPairData->mapMapSam[pid]);
			}*/
		}
		for (auto pair : pCurrSegInstance)
		{
			auto cid = pair.first;
			if (cid == 0)
				continue;
			if (!tempIDs.count(cid))
				pPairData->setSegToFailed.insert(cid);
		}

		AssociateLocalMapWithUncertainty(SLAM, pPairData);
		//segtofailed�� ��Ī ���� �߰��ϱ�.
		//SAM ��û ������ recovery �ٷ� �߰��ϱ�.
		for (auto pair : pPairData->mapLocalMapSeg)
		{
			auto cid = pair.second;
			if (pPairData->setSegToFailed.count(cid))
			{
				auto pG = pPairData->mpLocalMapIns->GaussianMaps.Get(pair.first);
				pG->AddObservation(pCurrSegMask, pCurrSegInstance[cid]);
				pCurrSegMask->GaussianMaps.Update(cid, pG);
			}
		}

		if (ptdata.rows == 0)
		{
			//SAM�� ���� �� local map ����� �ٷ� �߰���.
			auto mapIns = pPairData->mapMapSeg;
			/*for (auto pair : mapIns)
			{
				auto pid = pair.first;
				auto cid = pair.second;
				auto pins = pPairData->mpPrevMapIns->FrameInstances.Get(pid);

				if (pPairData->setSegFromFailed.count(pid))
				{
					pPairData->mapRaftSeg[pid] = cid;
				}
			}*/

			for (auto pair : pPairData->mapMapSam)
			{

				auto pid = pair.first;
				auto cid = pair.second;

				//if (pPairData->mapPrevMapResult[pid] == AssociationResultType::REPLACE)
				//{
				//	auto pins = pPairData->mpPrevMapIns->FrameInstances.Get(pid);
				//}

				if (pPairData->mapPrevMapResult[pid] == AssociationResultType::RECOVERY)
				{
					auto pins = pPairData->mpPrevMapIns->FrameInstances.Get(pid);
					int nNewID = AddNewInstance(pCurrSegMask, pins);

					pPairData->mapRaftSeg[pid] = nNewID;
					pPairData->mapRaftResult[pid] = AssociationResultType::SUCCESS;
				}

				/*auto pins = pPairData->mpPrevMapIns->FrameInstances.Get(pid);

				if (pPairData->setSegFromFailed.count(pid))
				{
					int nNewID = AddNewInstance(pCurrSegMask, pins);
					pPairData->mapRaftSam[pid] = nNewID;
				}*/
			}

			//local map ����� ���� �����ӿ� �߰��ϴµ� �̰͵� �񱳰� �ʿ����� �ʳ�?
			for (auto pair : pPairData->mapLocalMapSam)
			{
				auto pid = pair.first;
				if (pPairData->mapLocalMapResult[pid] == AssociationResultType::RECOVERY)
				{
					auto pins = pPairData->mpLocalMapIns->FrameInstances.Get(pid);
					int nNewID = AddNewInstance(pCurrSegMask, pins);

					auto pG = pPairData->mpLocalMapIns->GaussianMaps.Get(pid);
					pG->AddObservation(pCurrSegMask, pins);
					pCurrSegMask->GaussianMaps.Update(nNewID, pG);

					//pPairData->mapRaftSeg[pid] = nNewID;
					//pPairData->mapRaftResult[pid] = AssociationResultType::SUCCESS;
				}
			}
		}

		std::chrono::high_resolution_clock::time_point t_end = std::chrono::high_resolution_clock::now();
		auto du_seg = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
		//std::cout << "local map tine = " << du_seg << std::endl;
		//update gaussian object
		//���� ����ũ�� �߰��� ���뿡 ���ؼ��� ����
		//prev�� curr�̾�� ��. raft�� �ȵ�.

		//didx = 6;
		//DebugAssoSeg.Update(didx, DebugAssoSeg.Get(didx) + 1);

		UpdateGaussianObjectMap(pPairData->mapRaftSeg, pPrevSegMask, pCurrSegMask, InstanceType::SEG);
		if (mbObjBaselineTest)
		{
			UpdateBaselineObjectMap(pPairData->mapRaftSeg, pPrevSegMask, pCurrSegMask, InstanceType::SEG);

		}

		//regist object for visualization
		{
			auto mapGOs = pCurrSegMask->GaussianMaps.Get();
			for (auto pair : mapGOs)
			{
				auto pG = pair.second;
				if (!pG || !pG->mbInitialized)
					continue;
				cv::Mat data;
				pG->GenerateEllipsoidPoints(data);
				EdgeSLAM::Visualization::VisObject* pVisObj = nullptr;
				if (SLAM->mObjects.Count(pG->id))
					pVisObj = SLAM->mObjects.Get(pG->id);
				else
				{
					pVisObj = new EdgeSLAM::Visualization::VisObject();
					SLAM->mObjects.Update(pG->id, pVisObj);
					pVisObj->id = pG->id;
				}
				pVisObj->SetData(data);
				//std::cout <<pG->id<<" " << data.rows << std::endl;
			}
		}

		//didx = 7;
		//DebugAssoSeg.Update(didx, DebugAssoSeg.Get(didx) + 1);

		/*std::chrono::high_resolution_clock::time_point t3 = std::chrono::high_resolution_clock::now();
		auto du_u = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		auto du_o = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

		std::cout << "test = " << du_u << " " << du_o << std::endl;*/

		////��Ŀ���� ����
		//from���� ���� �ȵ� ����
		//1) from rect�ȿ���
		cv::Mat cimg = pNewBF->img.clone();
		cv::Mat pimg = pPrevBF->img.clone();

		if (mbObjBaselineTest)
		{
			cv::Mat K = pCurrKF->K.clone();
			cv::Mat T = pCurrKF->GetPose();
			cv::Mat R = T.rowRange(0, 3).colRange(0, 3);
			cv::Mat t = T.rowRange(0, 3).col(3);
			auto mapGIs = pCurrSegMask->MapInstances.Get();
			for (auto pair : mapGIs)
			{
				auto p = pair.second;
				if (!p)
					continue;
				std::vector<cv::Mat> vecMat;
				p->Update(vecMat);
				auto g2d = p->Project2D(K, R, t);
				auto rrect = g2d.CalcEllipse(1.0);
				
				cv::ellipse(cimg, rrect, cv::Scalar(255, 0, 0), 2);

				std::vector<cv::Point2f> vecBBs;
				p->ProjectBB(vecBBs, K, T);
				p->DrawBB(cimg, vecBBs);

				/*if (pCurrSegMask->GaussianMaps.Count(pair.first))
				{
					auto pG = pCurrSegMask->GaussianMaps.Get(pair.first);
					if (pG && pG->mbInitialized)
					{
						pG->SetPosition(p->GetPosition());
					}
				}*/
			}
		}

		std::vector<std::pair<cv::Point2f, cv::Point2f>> vecMatch, vecMatch2;
		
		cv::Mat resImage;
		SLAM->VisualizeMatchingImage(resImage, pimg, cimg, vecMatch, mapName, 0, cv::Scalar(0, 255, 255));

		if (ptdata.rows == 0){
			//���� ���� ���
			//prev map data �߰�

			VisualizeAssociation(SLAM, pPairData, mapName, 2);
		}

		/*didx = 8;
		DebugAssoSeg.Update(didx, DebugAssoSeg.Get(didx) + 1);
				
		std::cout << "Debug test SEG = ";
		auto debug = DebugAssoSeg.Get();
		for (auto pair : debug)
		{
			std::cout << pair.first<<"="<<pair.second << "  ";
		}
		std::cout << std::endl;*/
	}

	void AssociationManager::AssociationWithSAM(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key
		, const std::string& mapName, const std::string& userName, const int _type
		, AssoFramePairData* pPairData, InstanceMask* pMask){
	
		if (!mbSAM)
			return;

		InstanceType type = (InstanceType)_type;
		if (type == InstanceType::SEG)
		{
			pPairData->mpSamIns = pMask;
			AssociationWithSAMfromSEG(SLAM, ObjSLAM, key, mapName, userName, type, pPairData);
		}
		else if (type == InstanceType::MAP)
		{
			pPairData->mpSamIns2 = pMask;
			AssociationWithSAMfromMAP(SLAM, ObjSLAM, key, mapName, userName, type, pPairData);
		}

	}
	void AssociationManager::AssociationWithSAMfromMAP(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key
		, const std::string& mapName, const std::string& userName, const InstanceType& _type
		, AssoFramePairData* pPairData){
		
		std::cout << "MAP TEST ~~~~~~~~~~~" <<pPairData->toid<< std::endl;

		//SAM �����Ͱ� �ڱ�鳢�� ��ġ�ų�
		auto mapSamInstance = pPairData->mpSamIns2->FrameInstances.Get();
		auto mapFrameMapInstance = pPairData->mpFrameMapIns->FrameInstances.Get();

		std::map<std::pair<int, int>, AssoMatchRes*> res, mapSuccess;
		std::map<int, int> mnMatchedIdx;
		CalculateIOU(mapFrameMapInstance, mapSamInstance, res);
		EvaluateMatchResults(res, mapSuccess);

		int nTest = 0;
		for (auto pair : mapSuccess)
		{
			auto pid = pair.first.first;
			auto sid = pair.first.second;

			auto ares = pair.second;

			if (ares->iou > 0.5)
			{
				ares->res = true;
				//�� �ν��Ͻ��� �ߺ��� �� ����. Ȯ�� �ʿ�
				mnMatchedIdx[sid] = pid;
				nTest++;
			}
		}
		std::cout << "asdf = " << nTest << " " << mapFrameMapInstance.size() << std::endl;
	}

	void AssociationManager::AssociationWithSAMfromSEG(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key
		, const std::string& mapName, const std::string& userName, const InstanceType& _type
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
				//�� �ν��Ͻ��� �ߺ��� �� ����. Ȯ�� �ʿ�
				mnMatchedIdx[sid] = pid;
			}
		}

		//SAM���� �߰��� ���ο� �ν��Ͻ��� ����
		int nStartIdx = -1;
		int nLastIdx = -2;
		bool bInit = true;

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
				pPairData->mapRaftResult[pid] = AssociationResultType::SUCCESS;

				nLastIdx = nNewID;
				if (bInit)
				{
					bInit = false;
					nStartIdx = nNewID;
				}
			}
			else {
				pPairData->setSamToFailed.insert(sid);
			}
		}

		//���� recovery�� ���� fail�� ó��
		//fail �߿��� map�� ��Ī�� �� �ֵ��� success �� id �߰�
		for (auto pair : mapRaftInstance)
		{
			auto pid = pair.first;
			if (pPairData->mapRaftResult[pid] == AssociationResultType::RECOVERY)
			{
				//���� prev �Ӹ��� �ƴ� local map���� �ؾ� ��.
				auto temp = pPairData->mapPrevMapResult[pid];
				if (temp == AssociationResultType::SUCCESS)
				{
					//success �̸� �̹� �����ϴ� cid�� �ִ� ��. �߰��ϸ� �ȵ�.
					int cid = pPairData->mapMapSeg[pid];
					pPairData->mapRaftSam[pid] = cid;
					pPairData->mapRaftResult[pid] = AssociationResultType::SUCCESS;
				}
				if (temp == AssociationResultType::RECOVERY)
				{
					//RECOVERY�ε� SAM�� �ȵ� ��쿡 �� �����͸� �߰��ϱ�
					auto pins = pPairData->mpPrevMapIns->FrameInstances.Get(pid);
					int nNewID = AddNewInstance(pCurrSegMask, pins);
					pPairData->mapRaftSam[pid] = nNewID;
					pPairData->mapRaftResult[pid] = AssociationResultType::SUCCESS;
				}
			}
		}

		for (auto pair2 : pPairData->mapLocalMapSam)
		{
			auto pgid = pair2.first;
			auto pins = pPairData->mpLocalMapIns->FrameInstances.Get(pgid);
			auto pG = pPairData->mpLocalMapIns->GaussianMaps.Get(pgid);

			bool biou = true;

			for (int i = nStartIdx; i <= nLastIdx; i++)
			{
				auto cins = pCurrSegMask->FrameInstances.Get(i);
				float iou = CalculateIOU(pins->mask, cins->mask, pins->area, 1);

				if (iou > 0.5)
				{
					biou = false;
					pCurrSegMask->GaussianMaps.Update(i, pG);
					pG->AddObservation(pCurrSegMask, cins);
					//pG�� ����
					break;
				}
			}

			if (biou)
			{
				//instance�� pG �߰�
				int nNewID = AddNewInstance(pCurrSegMask, pins);
				pCurrSegMask->GaussianMaps.Update(nNewID, pG);
				pG->AddObservation(pCurrSegMask, pins);
			}

			/*float iou = CalculateIOU(pUins->mask, cins->mask, pUins->area, 1);
			if (iou > 0.5)
			{
				pPairData->mapLocalMapSeg[pG->id] = cid;
				pPairData->mapLocalMapResult[pG->id] = AssociationResultType::SUCCESS;

				cv::rectangle(cimg, pUins->rect, cv::Scalar(0, 255, 0), 2);

				bReqSam = false;
				break;
			}*/
		}

		//for (auto pair : mapRaftInstance)
		//{
		//	auto pid = pair.first;
		//	if (pPairData->mapRaftResult[pid] == AssociationResultType::SUCCESS)
		//	{
		//		bool biou = false;
		//		
		//		//���� prev �Ӹ��� �ƴ� local map���� �ؾ� ��.
		//		auto temp = pPairData->mapPrevMapResult[pid];
		//		if (temp == AssociationResultType::SUCCESS)
		//		{
		//			//success �̸� �̹� �����ϴ� cid�� �ִ� ��. �߰��ϸ� �ȵ�.
		//			int cid = pPairData->mapMapSeg[pid];
		//			pPairData->mapRaftSam[pid] = cid;
		//			pPairData->mapRaftResult[pid] = AssociationResultType::SUCCESS;
		//		}
		//		if (temp == AssociationResultType::RECOVERY)
		//		{
		//			//RECOVERY�ε� SAM�� �ȵ� ��쿡 �� �����͸� �߰��ϱ�
		//			auto pins = pPairData->mpPrevMapIns->FrameInstances.Get(pid);
		//			int nNewID = AddNewInstance(pCurrSegMask, pins);
		//			pPairData->mapRaftSam[pid] = nNewID;
		//			pPairData->mapRaftResult[pid] = AssociationResultType::SUCCESS;
		//		}
		//	}
		//}

		for (auto pair : pPairData->mapReqRaft)
		{
			auto pid = pair.first;
			if (!pPairData->mapRaftSam.count(pid))
				pPairData->setSamFromFailed.insert(pid);
		}

		//prev map ��Ī �� sam �߰� �ȵ� �� �߰�

		UpdateGaussianObjectMap(pPairData->mapRaftSam, pPrevSegMask, pCurrSegMask, InstanceType::SEG);
		
		if (mbObjBaselineTest)
		{
			UpdateBaselineObjectMap(pPairData->mapRaftSam, pPrevSegMask, pCurrSegMask, InstanceType::SEG);
		}
		
		VisualizeAssociation(SLAM, pPairData, mapName, 2, 1);

		//TestUncertainty(SLAM, ObjSLAM, pPairData);

	}
	void AssociationManager::AssociateLocalMapWithUncertainty(EdgeSLAM::SLAM* SLAM, AssoFramePairData* pPairData) {
		//������Ʈ ����
		////1. local map
		////2. �׸��� + ī�޶󿡼� �׸��� ���̸�(V)
		////3. ��ü ���� ����
		//����
		////1. prev frame�� �ش� ��ü�� ������ ���� ����
		////2. prev�� ���Ե� ��ü�� ����� ��ü��
		////���������� SAM��û����
		/*int didx = 0;
		didx = 0;
		DebugAssoSAM.Update(didx, DebugAssoSAM.Get(didx) + 1);*/

		std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();

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

		int margin = 1;
		int w2 = w - margin;
		int h2 = h - margin;

		pPairData->mpLocalMapIns = new InstanceMask();
		pPairData->mpLocalMapIns->id1 = pPairData->toid; //target, current
		pPairData->mpLocalMapIns->id2 = pPairData->fromid;//reference, previous

		std::map<int, GOMAP::GaussianObject*> mpGOs;
		GetLocalObjectMaps(pPrevSegMask, mpGOs); 

		//���� �����ӿ��� ��Ī ������ ��ü�� �߰�
		auto pPrevGO = pPrevSegMask->GaussianMaps.Get();
		for (auto pair : pPrevGO)
		{
			auto pid = pair.first;
			auto pG = pair.second;
			if (!pG || !pG->mbInitialized)
				continue;
			if (pPairData->mapPrevMapResult[pid] == AssociationResultType::FAIL)
			{
				if (!mpGOs.count(pG->id))
					mpGOs[pG->id] = pG;
			}
		}

		//��������
		auto mapCurrIns = pCurrSegMask->FrameInstances.Get();
		//std::map<int, FrameInstance*> mapFailedCurrIns;
		std::map<int, int> mapTempMatches;
		//std::map<int, AssoMatchRes*> mapAssoRes;//id1 : �ڱ��ڽ�, id2 : �۷ι� ��ü id
		std::map<int, std::set<GOMAP::GaussianObject*>> mapAssoGOs;

		cv::Mat cimg = pCurrBF->img.clone();

		/*for (auto cid : pPairData->setSegToFailed)
		{
			mapFailedCurrIns[cid] = pCurrSegMask->FrameInstances.Get(cid);
			auto ares = new AssoMatchRes();
			ares->id1 = cid;
			mapAssoRes[cid] = ares;
			cv::rectangle(cimg, mapFailedCurrIns[cid]->rect, cv::Scalar(0, 0, 255), 2);
		}*/

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

			//�����̶� ��ġ�°� �־�� �ϵ��� �߾���.
			std::vector<int> vecCurrInsIdx;
			for (auto pair2 : mapCurrIns)
			{
				auto cid = pair2.first;
				auto cins = pair2.second;

				if (cid == 0)
					continue;

				float iou = morf.map2D.CalcIOU(cins->rect);
				if (iou > 0.0)
				{
					vecCurrInsIdx.push_back(cid);
					//cv::ellipse(cimg, morf.region, cv::Scalar(0, 255, 0), 2);
				}
			}

			//currKF���� map�� ������ ������
			ExtractRegionFeatures(pCurrKF, morf.region, morf.keypoints, morf.mappoints, morf.descriptors);

			//���� ��Ī ���� ���� �������� ����
			int nobsidx = 0;
			std::map<int, ObjectRegionFeatures> mapTempFeatures;
			std::map<int, std::vector<std::pair<int, int>> > mapTempMatches;
			std::vector < std::pair<int, int>> vecTempPairs;

			//obs frame region feature
			auto obs = pG->mObservations.Get();
			for (auto pair2 : obs)
			{
				auto pIns = pair2.second;
				auto pTempKF = pIns->mpRefKF;

				//�������̼� �����ӿ��� �ٿ�� �ڽ� ���� Ư¡ ����
				ObjectRegionFeatures porf(pG, pIns);
				porf.rect = porf.mpRefIns->rect;
				ExtractRegionFeatures(pTempKF, porf.rect, porf.keypoints, porf.mappoints, porf.descriptors);

				//�������̼��� ������ �ʿ��� ���������� ���� Ư¡�� curr�� ���� ���� ��Ī
				std::vector<std::pair<int, int>> vecMatches, vecFilteredMatches, vecResMatches;
				ObjectMatcher::SearchInstance(porf.descriptors, morf.descriptors, vecMatches);
				bool bNew = ObjectMatcher::removeOutliersWithMahalanobis(porf.keypoints, morf.keypoints, vecMatches, vecFilteredMatches);

				if (bNew)
				{
					mapTempFeatures[nobsidx] = porf;
					mapTempMatches[nobsidx] = vecFilteredMatches;
					vecTempPairs.push_back(std::make_pair((int)vecFilteredMatches.size(), nobsidx++));
				}

				//if (bNew) {
				//	//std::cout << "matching test = " << pG->id << " " << vecFilteredMatches.size() << " " << bNew << std::endl;
				//	//��Ī�� ������ ��� �������̼��� ���� Ư¡�� curr KF�� ǥ��
				//	auto pUins = GenerateFrameInsWithUncertainty(pCurrKF, porf, morf, vecFilteredMatches);
				//	
				//	cv::rectangle(cimg, pUins->rect, cv::Scalar(255, 255, 0), 3);

				//	//curr kf�� ǥ���� �ʿ��� �����Ǵ� �������̼��� �ٿ�� �ڽ��� ��Ī�ϴ� �ڽ� ã��.
				//	//������ ������ �߰�
				//	//���⼭ �߰��Ǵ� ���� ������ ���� �� �־ �̰� üũ ���ؾ� ��.
				//	for (auto cid : vecTempIDXs)
				//	{
				//		auto cins = mapFailedCurrIns[cid];
				//		float iou = CalculateIOU(pUins->mask, cins->mask, pUins->area, 1);
				//		auto ares = mapAssoRes[cid];
				//		if (iou > 0.5)
				//		{  
				//			ares->res = true;
				//			if (ares->iou < iou)
				//			{
				//				ares->iou = iou;
				//				ares->id2 = pG->id;

				//			}
				//			//pPairData->mapRaftSeg[pid] = pair2.first;
				//			if (!mapAssoGOs[cid].count(pG))
				//				mapAssoGOs[cid].insert(pG);
				//		}
				//	}
				//}

			}//observation
			
			std::sort(vecTempPairs.begin(), vecTempPairs.end(), std::greater<>());
			//���ļ����� ���� ��Ī �����ϸ� �극��ũ.
			//�ƴϸ� ù��° �ν��Ͻ��� �ϴ� �߰�
			bool bReqSam = true;
			/*for (auto pair : vecTempPairs)
			{
				auto tid = pair.second;
				auto pUins = GenerateFrameInsWithUncertainty(pCurrKF, mapTempFeatures[tid], morf, mapTempMatches[tid]);
				for (int cid : vecCurrInsIdx)
				{
					auto cins = mapCurrIns[cid];
					float iou = CalculateIOU(pUins->mask, cins->mask, pUins->area, 1);
					if (iou > 0.5)
					{
						pPairData->mapLocalMapSeg[tid] = cid;
						pPairData->mpLocalMapIns->FrameInstances.Update(tid, pUins);
						pPairData->mapLocalMapResult[tid] = AssociationResultType::SUCCESS;

						bSeg = true;
						break;
					}
				}
				
				break;
			}*/
			if (vecTempPairs.size() > 0)
			{
				int tid = vecTempPairs[0].second;
				auto pUins = GenerateFrameInsWithUncertainty(pCurrKF, mapTempFeatures[tid], morf, mapTempMatches[tid]);
				pPairData->mpLocalMapIns->FrameInstances.Update(pG->id, pUins);
				pPairData->mpLocalMapIns->GaussianMaps.Update(pG->id, pG);

				cv::rectangle(cimg, pUins->rect, cv::Scalar(0, 255, 255), 1);

				//���� ���� �� �߿��� ��ġ���� Ȯ��
				for (auto pair3 : pPrevGO)
				{
					auto pG2 = pair3.second;
					if (!pG2 || !pG2->mbInitialized)
						continue;
					if (pG == pG2)
						continue;
					if (pG2->IsSameObject(pG))
					{
						cv::rectangle(cimg, pUins->rect, cv::Scalar(255, 255, 0), 5);
					}
				}

				for (int cid : vecCurrInsIdx)
				{
					auto cins = mapCurrIns[cid];

					float iou = CalculateIOU(pUins->mask, cins->mask, pUins->area, 1);
					if (iou > 0.5)
					{
						pPairData->mapLocalMapSeg[pG->id] = cid;
						pPairData->mapLocalMapResult[pG->id] = AssociationResultType::SUCCESS;

						cv::rectangle(cimg, pUins->rect, cv::Scalar(0, 255, 0), 2);

						bReqSam = false;
						break;
					}
				}
				if (bReqSam)
				{
					pPairData->mapLocalMapSam[pG->id] = 0;
					pPairData->mapLocalMapResult[pG->id] = AssociationResultType::RECOVERY;
				}
			}
		}

		//��� üũ

		//������Ʈ ���� ������

		//��Ȯ�Ǽ����� ���� ���� ����
		
		//������ �������̼����� ������ ���� Ư¡ ����
		////��� �����Ӱ� �ϴ� ���ϰ� �ϰ� ������ ������ ������ �ʿ���.
				
		////��Ī  
		//for (auto pair : mapAssoRes)
		//{
		//	auto cid = pair.first;

		//	auto ares = pair.second;
		//	if (!ares->res)
		//		continue;

		//	auto spGOs = mapAssoGOs[cid];
		//	auto pG = mpGOs[ares->id2];
		//	if (spGOs.size() > 1)
		//	{
		//		std::cout << cid << " == " << spGOs.size() << std::endl;
		//	}
		//	cv::rectangle(cimg, mapFailedCurrIns[cid]->rect, cv::Scalar(0, 255, 0), 2);
		//	pCurrSegMask->GaussianMaps.Update(cid, pG);
		//}

		/*std::cout << "Debug Local Map = ";
		didx = 1;
		DebugAssoSAM.Update(didx, DebugAssoSAM.Get(didx) + 1);

		auto debug = DebugAssoSAM.Get();
		for (auto pair : debug)
		{
			std::cout << pair.first << "=" << pair.second << "  ";
		}
		std::cout << std::endl;*/

		std::chrono::high_resolution_clock::time_point t_end = std::chrono::high_resolution_clock::now();
		auto du_seg = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
		std::cout << "local map duration = " << du_seg << std::endl;

		std::stringstream ss;
		ss.str("");
		ss << "../res/asso/" << pPairData->toid << "_" << pPairData->fromid << "_" << 3 << ".png";
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

		//���� �������� ���鼭 ���� �����ӿ� ��������
		for (auto pBF : vpBFs)
		{
			auto pKF = pBF->mpRefKF;
			auto pSegMask = pBF->mapMasks.Get("yoloseg");
			auto mapTempGOs = pSegMask->GaussianMaps.Get();

			cv::Mat pimg = pBF->img.clone();
			cv::Mat cimg = pCurrBF->img.clone();

			std::vector<std::pair<cv::Point2f, cv::Point2f>> vecTestSuccess,  vecTestFailed;
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
			//���
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

	bool AssociationManager::AssociationWithFrame(BoxFrame* pCurrBF, EdgeSLAM::Frame* pFrame,
		const cv::Mat& Kc, const cv::Mat& T, std::map<GOMAP::GaussianObject*, FrameInstance*>& mapRes) {
		
		if (!pCurrBF)
			return false;
		if (!pCurrBF->mapMasks.Count("yoloseg"))
			return false;

		auto pCurrSegMask = pCurrBF->mapMasks.Get("yoloseg");
		auto pCurrKF = pCurrBF->mpRefKF;

		const cv::Mat Rc = T.rowRange(0, 3).colRange(0, 3);
		const cv::Mat tc = T.rowRange(0, 3).col(3);

		float chi = sqrt(5.991);

		int w = pCurrKF->mpCamera->mnWidth;
		int h = pCurrKF->mpCamera->mnHeight;

		auto mapGOs = pCurrSegMask->GaussianMaps.Get();
		int n = 0;
		int n2 = 0;
		int n3 = 0;
		int n4 = 0;
		for (auto pair : mapGOs)
		{
			auto pid = pair.first;
			auto pG = pair.second;

			if (pG && pG->mbInitialized) {
				auto pins = pCurrSegMask->FrameInstances.Get(pid);
				ObjectRegionFeatures porf(nullptr, pins);
				porf.rect = porf.mpRefIns->rect;
				ExtractRegionFeatures(pCurrKF, porf.rect, porf.keypoints, porf.descriptors);

				ObjectRegionFeatures morf(pG, pins);
				morf.map2D = morf.mpRefMap->Project2D(Kc, Rc, tc);
				morf.region = morf.map2D.CalcEllipse(chi);
				morf.rect = morf.region.boundingRect();
				ExtractRegionFeatures(pFrame, morf.region, morf.keypoints, morf.descriptors);

				//�� region feaeture ��
				std::vector<std::pair<int, int>> vecMatches, vecFilteredMatches, vecResMatches;
				ObjectMatcher::SearchInstance(porf.descriptors, morf.descriptors, vecMatches);
				bool bNew = ObjectMatcher::removeOutliersWithMahalanobis(porf.keypoints, morf.keypoints, vecMatches, vecFilteredMatches);

				//���ο� �ν��Ͻ� ����.
				//�� �ν��Ͻ��� ���ο� �����Ӱ� ��
				if (bNew)
				{
					n++;
					//�� ����ũ�� ���� 
					auto pUins = GenerateFrameInsWithUncertainty(nullptr, porf, morf, vecFilteredMatches);
					mapRes[pG] = pUins;
					//GO�� FrameInstance�� �Ѱܾ� ��.
				}
				else {
					n2++;
				}
			}
			else {
				if(!pG)
					n3++;
				if (pG && !pG->mbInitialized)
					n4++;
			}
		}

		std::cout << "obj test = " << n <<" "<<n2 <<" "<<n3 <<" "<<n4 << " || " << mapGOs.size() << std::endl;
		return true;
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

				//�� region feaeture ��
				std::vector<std::pair<int, int>> vecMatches, vecFilteredMatches, vecResMatches;
				ObjectMatcher::SearchInstance(porf.descriptors, morf.descriptors, vecMatches);
				bool bNew = ObjectMatcher::removeOutliersWithMahalanobis(porf.keypoints, morf.keypoints, vecMatches, vecFilteredMatches);

				//���ο� �ν��Ͻ� ����.
				//�� �ν��Ͻ��� ���ο� �����Ӱ� ��
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

	void AssociationManager::AssociationPrevMapWithUncertainty(EdgeSLAM::SLAM* SLAM, AssoFramePairData* pPairData) 
	{
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

		std::map<int, FrameInstance*> mapCurrIns = pCurrSegMask->FrameInstances.Get();
		auto mapPrevIns = pPrevSegMask->FrameInstances.Get();

		auto mapPrevGOs = pPrevSegMask->GaussianMaps.Get();

		pPairData->mpPrevMapIns = new InstanceMask();
		pPairData->mpPrevMapIns->id1 = pPairData->toid; //target, current
		pPairData->mpPrevMapIns->id2 = pPairData->fromid;//reference, previous

		for (auto pair : mapPrevGOs)
		{
			auto pid = pair.first;
			auto pG = pair.second;

			if (pid == 0)
				continue;
			if (!pG || !pG->mbInitialized)
			{
				continue;
			}
			//prev region
			
			auto pins = pPrevSegMask->FrameInstances.Get(pid);
			ObjectRegionFeatures porf(nullptr, pins);
			porf.rect = porf.mpRefIns->rect;
			ExtractRegionFeatures(pPrevKF, porf.rect, porf.keypoints, porf.descriptors);

			//curr uncertainty region
			ObjectRegionFeatures morf(pG, pins);
			morf.map2D = morf.mpRefMap->Project2D(Kc, Rc, tc);
			morf.region = morf.map2D.CalcEllipse(chi);
			morf.rect = morf.region.boundingRect();
			ExtractRegionFeatures(pCurrKF, morf.region, morf.keypoints, morf.descriptors);

			//�� region feaeture ��
			std::vector<std::pair<int, int>> vecMatches, vecFilteredMatches, vecResMatches;
			ObjectMatcher::SearchInstance(porf.descriptors, morf.descriptors, vecMatches);
			bool bNew = ObjectMatcher::removeOutliersWithMahalanobis(porf.keypoints, morf.keypoints, vecMatches, vecFilteredMatches);

			//���ο� �ν��Ͻ� ����.
			//�� �ν��Ͻ��� ���ο� �����Ӱ� ��
			if (bNew)
			{
				auto pUins = GenerateFrameInsWithUncertainty(pCurrKF, porf, morf, vecFilteredMatches);
				pPairData->mpPrevMapIns->FrameInstances.Update(pid, pUins);

				for (auto pair2 : mapCurrIns)
				{
					auto cid = pair2.first;
					
					auto cins = pair2.second;
					ObjectMeasureType type;
					float iou = CalculateMeasureForAsso(pUins->mask, cins->mask, pUins->area, cins->area, type, cid);
					if (iou > 0.5)
					{
						AssociationResultType res;
						if (cid == 0)
						{
							pPairData->mapMapSam[pid] = 0;
							res = AssociationResultType::RECOVERY;
						}
						else {
							/*if (type == ObjectMeasureType::IoM)
							{
								if (pUins->area > cins->area)
								{
									res = AssociationResultType::REPLACE;
									pPairData->mapMapSeg[pid] = cid;
								}
							}
							else {
								res = AssociationResultType::SUCCESS;
								pPairData->mapMapSeg[pid] = cid;
							}*/
							res = AssociationResultType::SUCCESS;
							pPairData->mapMapSeg[pid] = cid;
						}
						pPairData->mapPrevMapResult[pid] = res;
						/*pPairData->mapRaftSeg[pid] = cid;
						pPairData->setSegFromFailed.erase(pid);
						pPairData->setSegToFailed.erase(cid);*/
					}
				}
			}
		}//for map

		//������ �������� ���� ��� �ν��Ͻ��� fail �߰�
		for (auto pair : mapPrevIns)
		{
			int pid = pair.first;
			if (!pPairData->mapPrevMapResult.count(pid))
			{
				pPairData->mapPrevMapResult[pid] = AssociationResultType::FAIL;
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
		
		//�ν��Ͻ�
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

		//����Ʈ ����
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

		//����Ʈ ��Ī
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

		//�� ���̵� ����



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

		//�ð�ȭ
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
			//���� �׽�Ʈ �ϴ� ��
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
	//u�� ���ʸ��̸� IoA, ���̸� IoU, �ּҰ��̸� IoM
	float AssociationManager::CalculateMeasureForAsso(const cv::Mat& mask1, const cv::Mat& mask2, float area1, float area2, ObjectMeasureType& type, int id) {

		cv::Mat overlap = mask1 & mask2;
		float nOverlap = (float)cv::countNonZero(overlap);
		float iou = 0.0;
		float th = 0.5;
		type = ObjectMeasureType::IoU;
		if (nOverlap == 0.0)
		{

		}
		else {
			if (id == 0)
			{
				//ioa
				type = ObjectMeasureType::IoA;
				iou = CalculateIOU(nOverlap, area1);
			}
			else {
				cv::Mat total = mask1 | mask2;
				float nUnion = (float)cv::countNonZero(total);
				iou = CalculateIOU(nOverlap, nUnion);
				if (iou < th)
				{
					type = ObjectMeasureType::IoM;
					float nMin = std::min(area1, area2);
					iou = CalculateIOU(nOverlap, nMin);
				}
			}
		}
		return iou;
	}
	float AssociationManager::CalculateIOU(const float& i, const float& u) {
		return i / u;
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

				////��ġ�°� ������ ����
				//if (nOverlap == 0) {
				//	ares->iou = 0.0;
				//}
				//else {
				//	float iou = 0.0;

				//	//��׶���� ������, ������ �ν��Ͻ��� �������� ���� �ٸ�.
				//	//��׶���� �񱳽� SAM ��û ����
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

		//prev���� �ߺ��� �����ϰ�
		//�װ� �������� curr���� ��Ī�� ����.

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
		//���� ������ ��ü �����κ��� ������ ������ ���� ȹ��
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
		////��������
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
			pNewIns->mask = cv::Mat::zeros(h, w, CV_8UC1); //ellipse�� ��ü�ϴ���
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
		//��ü �� ������ �ּ�ȭ�ϴ� ��
		//���� ������ ������ ��ġ�� ���� ��û�ϰ� �ϴ� ���� ���� ��
		//��ü�� �ٷ� �����ϵ��� ����
		std::set<GOMAP::GaussianObject*> setGOs;
		std::set<InstanceMask*> setFrames;

		////Ű�����ӿ��� �ĺ��� ����
		auto mapPrevGOs = pPrevSeg->GaussianMaps.Get();
		//���� ������ ��ü �����κ��� ������ ������ ���� ȹ��
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

		//���� ���������κ��� ��ü ���� �߰�
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
		////Ű�����ӿ��� �ĺ��� ����

		////��������
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
		////��������

		////�ߺ� üũ
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

		//FrameInstance �߰�
		//���� �����ӿ� �ν��Ͻ��� ����ǰ�, RAFT�� ��ȯ�� �Ǹ� �ش� �ν��Ͻ��� ���������� ��ȯ��.
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
		//FrameInstance �߰�


		////2����, 3���� ����

		//A : ���� �����ӿ� ����� �ֵ� ����

		//B : ���� �����ӿ� ����� �ֵ� �� �� ��û�� �ֵ� ����

		//C : �������� �� �̹��� ���� �ٱ��� ��ġ�� ������Ʈ ����
		////�� �ȿ� �����ϴ� ������ üũ

		//���� ����

		//A,B,C�� �����ؼ� ��Ƴ��� ��ü�� ����

		////���� Ȯ��
		//2d iou�� ��ħ Ȯ��
		//3d ���Ҷ��� �Ÿ��� ���� üũ

		//������ �ν��Ͻ��� ��ȯ, �������� id�� �����ص� ��.
	}

	void AssociationManager::RequestSAM(BoxFrame* pNewBF, std::map<int, FrameInstance*>& mapMaskInstance, 
		const std::map<GlobalInstance*, cv::Rect>& mapGlobalRect, 
		const std::vector<AssoMatchRes*>& vecResAsso, 
		int id1, int id2, const std::string& userName) {

		//�������� ���� �ʿ�.
		//������Ʈ ���� �ν��Ͻ�ȭ ��Ŵ

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
		//		//���� ���
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

		////�۷ι� ������Ʈ ó�� �ʿ�
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
	//baseline updatee
	void AssociationManager::UpdateBaselineObjectMap(std::map<int, int>& mapRes, InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, const InstanceType& type, bool bTest)
	{
		for (auto pair : mapRes)
		{
			auto pid = pair.first;
			auto cid = pair.second;

			auto pIns = pPrevSegMask->FrameInstances.Get(pid);
			auto cIns = pCurrSegMask->FrameInstances.Get(cid);

			auto pG1 = pPrevSegMask->MapInstances.Get(pid);
			auto pG2 = pCurrSegMask->MapInstances.Get(cid);

			//���� �ٸ��� ����

			//���� �ϳ��� ������ �ϳ� �߰�
			//�ƿ� ������ �⺻ ����
			//��ġ �ʱ�ȭ �ʿ��ϸ� �ʱ�ȭ

			GlobalInstance* pGO = nullptr;
			
			if (!pG1 && !pG2)
			{
				pGO = new GlobalInstance();
				pGO->Connect(pIns);
				pGO->Connect(cIns);

			}
			else if (pG1 && !pG2)
			{
				pG1->Connect(cIns);
				pGO = pG1;
				pGO->EIFFilterOutlier();
			}
			else if (pG2 && !pG1)
			{
				pG2->Connect(pIns);
				pGO = pG2;
				pGO->EIFFilterOutlier();
			}
			else if (pG1 && pG2)
			{
				if (pG1 != pG2)
				{
					std::cout << "update go error : ��ü1, ��ü2�� �ִµ� ���� �ٸ�!!! ���� �ʿ�" << std::endl;
				}
			}
			else {
				std::cout << "Update GO Map??????" << std::endl;
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
			//	std::cout << "update go error : ��ü2�� ����" << std::endl;
			//}
			//if (pG1 && pG2) {
			//	if (pG1 == pG2)
			//	{

			//	}
			//	
			//}
			////error check

			//������ ����
			//type == InstanceType::SEG && 
			if (pGO && !pG1)
			{
				pPrevSegMask->MapInstances.Update(pid, pGO);
			}
			if (pGO && !pG2)
			{
				pCurrSegMask->MapInstances.Update(cid, pGO);
			}

		}
	}
	//������Ʈ�� �и�
	void AssociationManager::UpdateGaussianObjectMap2(std::map<int, int>& mapRes, InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, const InstanceType& type)
	{
		/*long long total = 0;
		long long total2 = 0;
		int n = 0;*/
		//local map���� ������ ������ ������ ���
		//�����꿡 ������ ��ü�� ����.
		//�� ��ü�� ������ �ʱ�ȭ�Ǿ�����.

		for (auto pair : mapRes)
		{
			auto pid = pair.first;
			auto cid = pair.second;

			auto cIns = pCurrSegMask->FrameInstances.Get(cid);

			auto pG1 = pPrevSegMask->GaussianMaps.Get(pid);
			auto pG2 = pCurrSegMask->GaussianMaps.Get(cid);

			//���� �ٸ��� ����

			//���� �ϳ��� ������ �ϳ� �߰�
			//�ƿ� ������ �⺻ ����
			//��ġ �ʱ�ȭ �ʿ��ϸ� �ʱ�ȭ

			GOMAP::GaussianObject* pGO = nullptr;

			if (pG1 && !pG2)
			{
				pG1->AddObservation(pCurrSegMask, cIns);
				if (pG1->mbInitialized) {
					GOMAP::Optimizer::ObjectOptimizer::ObjectPosOptimization(pG1);
					GaussianMapManager::UpdateObjectWithIncremental(pG1, cIns);
				}
				pGO = pG1;
			}
			else if (pG1 && pG2)
			{
				if (pG1 != pG2)
				{
					std::cout << "update go error : ��ü1, ��ü2�� �ִµ� ���� �ٸ�!!! ���� �ʿ� == " << pG1->id << " " << pG2->id << " == " << pG1->CalcDistance3D(pG2) << std::endl;
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
			
			if (pGO && !pG2)
			{
				pCurrSegMask->GaussianMaps.Update(cid, pGO);
			}

		}

	}
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

			//���� �ٸ��� ����
			
			//���� �ϳ��� ������ �ϳ� �߰�
			//�ƿ� ������ �⺻ ����
			//��ġ �ʱ�ȭ �ʿ��ϸ� �ʱ�ȭ

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
					std::cout << "update go error : ��ü1, ��ü2�� �ִµ� ���� �ٸ�!!! ���� �ʿ� == " <<pG1->id<<" "<<pG2->id<<" == " << pG1->CalcDistance3D(pG2) << std::endl;
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
			//	std::cout << "update go error : ��ü2�� ����" << std::endl;
			//}
			//if (pG1 && pG2) {
			//	if (pG1 == pG2)
			//	{

			//	}
			//	
			//}
			////error check

			//������ ����
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
		//�ð�ȭ
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
		
		//�ð�ȭ
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

		//�ð�ȭ
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

			//��ġ�°� ������ ����
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

		//��׶��忡 �Ϻп� ���ϸ鼭
		//���� ����ũ�� �ϳ��� �Ȱ��ľ� ��.

		return bres1 & bres2;
	}

	int AssociationManager::AddNewInstance(InstanceMask* pFrame, FrameInstance* pNew) {
		int nNewID = pFrame->mnMaxId++;
		pFrame->FrameInstances.Update(nNewID, pNew);
		pFrame->GaussianMaps.Update(nNewID, nullptr);
		pFrame->MapInstances.Update(nNewID, nullptr);
		 
		//mapInstatnces[nNewID] = pNew;
		return nNewID;
		/*return;
		auto pBG = pFrame->FrameInstances.Get(0);
		cv::Mat maskBG = pBG->mask.clone();
		maskBG -= pNew->mask;*/

		//background update
		//�ν��Ͻ����� ��ü ����ũ �̿��ϴ��� Ȯ�� �ʿ�
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
		
	void AssociationManager::ExtractRegionFeatures(EdgeSLAM::Frame* pF, const cv::RotatedRect& rect, std::vector<cv::KeyPoint>& vecKPs, cv::Mat& desc){
		desc = cv::Mat::zeros(0, 32, CV_8UC1);

		for (int i = 0; i < pF->mvKeys.size(); i++) {
			auto kp = pF->mvKeys[i];
			if (isPointInRotatedRect(kp.pt, rect))
			{
				vecKPs.push_back(kp);
				desc.push_back(pF->mDescriptors.row(i));
			}
		}
	}
	void AssociationManager::ExtractRegionFeatures(EdgeSLAM::Frame* pF, const cv::RotatedRect& rect, 
		std::vector<cv::KeyPoint>& vecKPs, std::vector<EdgeSLAM::MapPoint*>& vecMPs, cv::Mat& desc){
		desc = cv::Mat::zeros(0, 32, CV_8UC1);

		for (int i = 0; i < pF->mvKeys.size(); i++) {
			auto kp = pF->mvKeys[i];
			if (isPointInRotatedRect(kp.pt, rect))
			{
				vecKPs.push_back(kp);
				desc.push_back(pF->mDescriptors.row(i));

				auto pMPi = pF->mvpMapPoints[i];
				if (pMPi && !pMPi->isBad())
				{
					vecMPs.push_back(pMPi);
				}
				else
					vecMPs.push_back(nullptr);
			}
		}
	}


	void AssociationManager::RequestSAMToServer(const std::map<int, FrameInstance*> mapRes, int currid, int previd, const std::string& user, InstanceType type) {
		cv::Mat ptdata(0, 1, CV_32FC1);

		for (auto pair : mapRes)
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
		//usernamee, id2(prev), type, tsrc, ptdata, curr id
		if (ptdata.rows > 0) {
			//reqest  
			int nobj = ptdata.rows;
			ptdata.push_back(cv::Mat::zeros(1500 - nobj, 1, CV_32FC1));
			//id2 : prev frame, type : (1) frame, (2) map
			std::string tsrc = user + ".Image." + std::to_string(nobj) + "." + std::to_string(previd) + "." + std::to_string((int)type);
			auto sam2key = "reqsam2";
			std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();
			auto du_upload = Utils::SendData(sam2key, tsrc, ptdata, currid, 15, t_start.time_since_epoch().count());
		}
	}

}
