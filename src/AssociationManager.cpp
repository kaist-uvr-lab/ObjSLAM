#include <AssociationManager.h>

//EdgeSLAM
#include <Utils.h>
#include <SLAM.h>
#include <Camera.h>
#include <KeyFrame.h>

//ObjectSLAM
#include <FrameInstance.h>
#include <GlobalInstance.h>
#include <BoxFrame.h>
#include <ObjectSLAM.h>
#include <ObjectMatcher.h>
#include <InstanceLinker.h>

//가우시안 객체 맵
#include <Gaussian/GaussianObject.h>
#include <Gaussian/GaussianMapManager.h>
#include <Gaussian/Visualizer.h>
#include <Gaussian/Optimization/ObjectOptimizer.h>

namespace ObjectSLAM {
	
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

	void AssociationManager::CalculateIOU(const std::map<int, FrameInstance*>& mapPrevInstance, const std::map<int, FrameInstance*>& mapCurrInstance
		, std::map<int, std::map<int, AssoMatchRes*>>& mapIOU, float th)
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

				auto currIns = pair2.second;

				AssoMatchRes* ares = new AssoMatchRes(id1, id2, prevIns->type, currIns->type);
				
				const cv::Mat cmask = pair2.second->mask;

				cv::Mat overlap = pmask & cmask;
				float nOverlap = (float)cv::countNonZero(overlap);

				//겹치는게 없으면 무시
				if (nOverlap == 0) {
					ares->iou = 0.0;
				}
				else {
					float iou = 0.0;

					//백그라운드와 비교인지, 추정된 인스턴스와 비교인지에 따라서 다름.
					//백그라운드와 비교시 SAM 요청 가능
					if (id2 == 0) {
						iou = nOverlap / area1;
					}
					if (id2 > 0)
					{
						cv::Mat total = pmask | cmask;
						float nUnion = (float)cv::countNonZero(total);
						iou = nOverlap / nUnion;
					}
					ares->iou = iou;
					
					/*if (iou > bestMatch.second)
					{
						bestMatch.first = id2;
						bestMatch.second = iou;
					}*/
				}
				mapIOU[id1][id2] = ares;
			}

		}
	}
	
	//void AssociationManager::AssociationWithSAM(InstanceMask* pCurrSeg, const std::map<int, FrameInstance*>& mapSamInstance, std::map<int, FrameInstance*>& mapMissingInstance, std::vector<AssoMatchRes*>& vecResAsso, float th) {
	//	//미싱 인스턴스와 이전 결과물을 연결
	//	std::map<int, AssoMatchRes*> mapResAsso;
	//	for (auto pair : mapMissingInstance)
	//	{
	//		for (auto asso : vecResAsso)
	//		{
	//			auto id1 = asso->id1;
	//			if (asso->nType1 == InstanceType::MAP)
	//				continue;
	//			if (pair.first == id1)
	//			{
	//				mapResAsso[id1] = asso;
	//				break;
	//			}
	//		}
	//	}

	//	std::set<int> sAlreadySegMatch; //프레임 미싱 인스턴스 중에서 매칭 된 애 기록
	//	std::set<int> sAlreadySamMatch;

	//	for (auto pair1 : mapSamInstance)
	//	{
	//		auto pSamIns = pair1.second;
	//		const cv::Mat sammask = pSamIns->mask;
	//		bool bres = false;
	//		//std::cout << "sama == " << pair1.first << std::endl;
	//		for (auto pair2 : mapMissingInstance) {

	//			auto pM = pair2.second;
	//			int pid = pair2.first;
	//			if (sAlreadySegMatch.count(pid))
	//				continue;
	//			auto pPrevIns = mapMissingInstance[pid];

	//			/*if(!pCurrSeg->mapResAsso.count(pid))
	//				std::cout<<"???????????????errerrerr"<<std::endl;
	//			auto assores = pCurrSeg->mapResAsso[pid];*/
	//			auto assores = mapResAsso[pid];

	//			if (!assores->req)
	//				std::cout << "???????????????errerrerr222222222" << std::endl;
	//			//assores->req = false;
	//			//assores->iou = 0.0;
	//			//std::cout <<pNewBF->mnId<<" "<<pPrevBF->mnId<<" == " << pid << " " << assores->id << " " << assores->res << " " << assores->req << std::endl;

	//			const cv::Mat raftmask = pM->mask;
	//			float iou = 0.0;

	//			//std::cout << "sama == a" << std::endl;

	//			int cid = assores->id2;
	//			if (cid == 0)
	//			{
	//				cv::Mat overlap = raftmask & sammask;
	//				float nOverlap = (float)cv::countNonZero(overlap);
	//				//iou = nOverlap / pM->area;
	//				cv::Mat total = raftmask | sammask;
	//				float nUnion = (float)cv::countNonZero(total);
	//				iou = nOverlap / nUnion;
	//			}
	//			
	//			//std::cout << "sama  == b" << std::endl;
	//			if (iou > th)
	//			{
	//				sAlreadySegMatch.insert(pid);
	//				sAlreadySamMatch.insert(pair1.first);
	//				assores->res = true;
	//				bres = true;
	//				//vecMatchPairs.push_back(std::make_pair(pid, cid));
	//				//cur matching
	//				
	//				cid = pCurrSeg->mnMaxId++;
	//				assores->id2 = cid;
	//				pCurrSeg->FrameInstances.Update(cid, pSamIns);
	//				pCurrSeg->MapInstances.Update(cid, nullptr);
	//				pCurrSeg->GaussianMaps.Update(cid, nullptr);
	//				pCurrSeg->mask += (sammask / 255) * cid;

	//			}
	//			else {
	//				//assores.id = cid;
	//				//assores.res = false;

	//			}
	//			//std::cout << "sama == c" << std::endl;
	//			assores->iou = iou;
	//			//pCurrSeg->mapResAsso[pid] = assores;

	//			if (bres)
	//			{
	//				break;
	//			}
	//		}
	//		//std::cout << "sama == " << pair1.first <<" end" << std::endl;
	//	}
	//}
	//
	////id에 0이 없으면 샘을 요청할 일이 없긴 함.
	//int AssociationManager::AssociationWithMask(const std::map<int, FrameInstance*>& mapMaskInstance, const std::map<int, FrameInstance*>& mapSourceInstance, std::vector<AssoMatchRes*>& mapAssoRes, float th) {
	//	
	//	std::set<int> sAlreadyFrameMatch;
	//	
	//	int nNeedSAM = 0;

	//	for (auto pair1 : mapMaskInstance) {
	//		int id1 = pair1.first;
	//		if (id1 == 0)
	//			continue;

	//		AssoMatchRes* assores = new AssoMatchRes();
	//		assores->id1 = id1;

	//		const cv::Mat pmask = pair1.second->mask;
	//		float area1 = (float)cv::countNonZero(pmask);

	//		//id, iou
	//		std::pair<int, float> bestFailMatch = std::make_pair(-1, 0.0);

	//		bool bres = false;

	//		for (auto pair2 : mapSourceInstance) {
	//			int id2 = pair2.first;
	//			//현재 인스턴스 중에서 이미 매칭이 되었으면 패스
	//			if (sAlreadyFrameMatch.count(id2))
	//				continue;
	//			const cv::Mat cmask = pair2.second->mask;

	//			cv::Mat overlap = pmask & cmask;
	//			float nOverlap = (float)cv::countNonZero(overlap);

	//			//겹치는게 없으면 무시
	//			if (nOverlap == 0)
	//				continue;

	//			float iou = 0.0;
	//			
	//			//백그라운드와 비교인지, 추정된 인스턴스와 비교인지에 따라서 다름.
	//			//백그라운드와 비교시 SAM 요청 가능
	//			if (id2 == 0) {
	//				iou = nOverlap / area1;
	//			}
	//			if (id2 > 0)
	//			{
	//				cv::Mat total = pmask | cmask;
	//				float nUnion = (float)cv::countNonZero(total);
	//				iou = nOverlap / nUnion;
	//			}

	//			if (iou >= th)
	//			{
	//				bres = true;

	//				assores->id2 = id2;
	//				assores->iou = iou;

	//				if (id2 == 0)
	//				{
	//					//인스턴스가 없어서 매칭이 안되는 경우임.
	//					//샘요청
	//					assores->res = false;
	//					assores->req = true;
	//				}
	//				else {
	//					//매칭 성공
	//					assores->id1 = id1;
	//					assores->res = true;
	//					sAlreadyFrameMatch.insert(id2);
	//				}
	//			}
	//			else {
	//				if (id2 > 0)
	//				{
	//					//매칭이 실패 중 겹치는 값이 있으면 이전 마스크를 복원해서 매칭이 될 수 있도록 함.
	//					if (iou > bestFailMatch.second)
	//					{
	//						bestFailMatch.first = id2;
	//						bestFailMatch.second = iou;
	//					}
	//				}
	//			}

	//			if (bres) {
	//				/*if (id2 > 0)
	//					vres += mask;*/
	//				break;
	//			}
	//		}

	//		//현재 인스턴스가 제대로 안되어있으면 복구하는 용도인데 이제 필요한가?
	//		if (!bres && bestFailMatch.first > 0)
	//		{
	//			//raft에서의 에러 또는 마스킹이 작을 때 생김.
	//			//확인 상 대부분이 마스킹만 잘되면 동작할 듯.
	//			assores->id2 = bestFailMatch.first;
	//			assores->res = false;
	//			assores->req = false;
	//			assores->iou = bestFailMatch.second;
	//		}

	//		//pCurrSegMask->mapResAsso[id1] = assores;
	//		assores->nType1 = pair1.second->type;

	//		//샘이 필요한지 확인
	//		if (assores->req)
	//			nNeedSAM++;
	//		mapAssoRes.push_back(assores);

	//	}
	//	//매칭 결과로 오브젝트 맵 연결 체크 필요함
	//	//현재 프레임에서 머 했는데 삭제함.
	//	return nNeedSAM;
	//}

	//void AssociationManager::AssociationWithMap() {
	//	//현재 프레임의 인스턴스에서 이전 프레임과
	//	//1) 매칭이 되었는지
	//	//2) 샘을 요청하였는지
	//	//키프레임 주변의 객체 맵에서
	//	//현재 뷰안에 존재하는 객체 맵에 대해서
	//	
	//	//1) 맵을 프레임에 프로젝션
	//	//- 2차원 위치와 2차원 타원 정보 : visualize2d
	//	//2) 뎁스 정보로 정렬
	//	//3) 맵과 맵 연결
	//	//4) 맵과 프레임 연결
	//	//뎁스 순으로 정렬하면서
	//	//프로젝션되는 4점 중 하나가 기존 사각형에 포함되면
	//	//중복이거나 가림으로 처리
	//	//가림 비율이 크면 무시. 작으면 계속 진행 
	//}
	void AssociationManager::GetLocalObjectMaps(InstanceMask* pMask, std::map<int,GOMAP::GaussianObject*>& spGOs) {
		auto mapPrevGOs = pMask->GaussianMaps.Get();
		std::set<InstanceMask*> setFrames;
		//이전 프레임 객체 정보로부터 인접한 프레임 정보 획득
		for (auto pair : mapPrevGOs)
		{
			auto pG = pair.second;
			if (!pG)
				continue;
			spGOs[pG->id] = pG;
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
				if (!pG)
					continue;
				if (!spGOs.count(pG->id)) {
					spGOs[pG->id] = pG;
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
			g.GetRect();
			spG2Ds[pG->id] = g;
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
			g.GetRect();
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
			g.GetRect();
			
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
			g.GetRect();
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
	void AssociationManager::AssociationWithSAM(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key
		, const int id, const int id2, const int _type
		, const std::string& mapName, const std::string& userName, BoxFrame* pNewBF 
		, InstanceMask* pCurrSegMask, std::map<int, FrameInstance*>& mapSamInstances, bool bShow)
	{
				

		auto pCurrSegInstance = pCurrSegMask->FrameInstances.Get();
		
		//맵 또는 프레임 정보 획득
		InstanceType type = (InstanceType)_type;
		std::string stype = "reqSAM";
		if (type == InstanceType::MAP)
			stype = "reqSAM_MAP";

		InstanceMask* pReqMask = nullptr;
		std::map<int, FrameInstance*> pReqSAMInstance;
		if (pNewBF->mapMasks.Count(stype))
		{
			pReqMask = pNewBF->mapMasks.Get(stype);
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
		std::map<int, std::map<int, AssoMatchRes*>> res; //map<pid, map<cid, asso>>
		std::set<int> mnAddCurrIdx, mnMatchedIdx;
		std::map<int, std::set<int>> mapMatchSAMnReq; //sam id, set<req id>
		std::map<int, std::set<FrameInstance*>> mapCurrMatchRes; //cid, previnstance
		CalculateIOU(pReqSAMInstance, mapSamInstances, res);

		auto pNewSamMask = InstanceMask();

		//인스턴스 비교용
		cv::Mat vimg = pNewBF->img.clone();
		for (auto pair : pCurrSegInstance)
		{
			auto pIns = pair.second;
			cv::rectangle(vimg, pIns->rect, cv::Scalar(0, 0, 255), -1);
			//cv::putText(vimg, std::to_string(pGO->id), pIns->pt, 2, 1.3, cv::Scalar(0, 0, 255), 2);
		}
		for (auto pair : pReqSAMInstance)
		{
			auto pIns = pair.second;
			auto rect = pIns->rect;

			cv::rectangle(vimg, rect, cv::Scalar(255, 0, 0), 5);
			//cv::putText(vimg, std::to_string(pGO->id), pIns->pt, 2, 1.3, cv::Scalar(0, 0, 255), 2);
		}
		for (auto pair : mapSamInstances)
		{
			auto pIns = pair.second;
			auto rect = pIns->rect;

			cv::rectangle(vimg, rect, cv::Scalar(255, 255, 0), 5);
		}

		//SAM & prev
		for (auto pair : res)
		{
			auto pid = pair.first;
			auto pReqIns = pReqSAMInstance[pid];

			for (auto pair2 : pair.second) {
				auto sid = pair2.first;
				auto ares = pair2.second;
				auto pSAMIns = mapSamInstances[sid];
				if (ares->iou > 0.5)
				{
					ares->res = true;
					//샘 인스턴스가 중복될 수 있음. 확인 필요
					mnMatchedIdx.insert(sid);
					mapMatchSAMnReq[sid].insert(pid);
					cv::rectangle(vimg, pSAMIns->rect, cv::Scalar(0, 255, 0), 5);
				}
			}
		}
		
		int tempVisID = 6;
		if (type == InstanceType::SEG)
			tempVisID++;
		
		//test visualization

		//SAM & CURR
		std::map<std::pair<int, int>, GOMAP::GaussianObject*> mapUpdatedGOs;
		std::map<int, FrameInstance*> tempPrevInstance = pReqSAMInstance;
		if (type == InstanceType::SEG)
			tempPrevInstance = pPrevSegInstance;
		for (auto pair : mapSamInstances)
		{
			auto sid = pair.first;
			auto pNewSAM = pair.second;

			if (CheckAddNewInstance(pCurrSegInstance, pNewSAM) && mnMatchedIdx.count(sid))
			{
				int nNewID = AddNewInstance(pCurrSegMask, pNewSAM);
				pCurrSegInstance[nNewID] = pNewSAM;
				//mnAddCurrIdx.insert(sid);
				if (mapMatchSAMnReq[sid].size() > 1)
				{
					std::cout << "SAM::Update - " << sid << " - asdf " << std::endl;
				}

				//이전 프레임 정보를 추가하는 것임.
				//현재 추가된 샘을 시각화하고
				for (auto pid : mapMatchSAMnReq[sid])
				{
					auto pReqIns = tempPrevInstance[pid];
					if (!pReqIns)
						std::cout <<pid<<" "<< tempPrevInstance.size() << "?????????????????????????" << std::endl << std::endl << std::endl;
					mapCurrMatchRes[nNewID].insert(pReqIns);

					auto pairkey = std::make_pair(pid, nNewID);
					
					mapUpdatedGOs[pairkey] = nullptr;

					auto rect = pNewSAM->rect;
					rect.width /= 2;
					rect.height /= 2;
					//rect.x += rect.width;
					rect.y += rect.width;
					cv::rectangle(vimg, rect, cv::Scalar(255, 0, 255), 5);
				}
			}
			else {
				auto rect = pNewSAM->rect;

				rect.width /= 2;
				rect.height /=  2;
				rect.x += rect.width;
				rect.y += rect.width;
				cv::rectangle(vimg, rect, cv::Scalar(0, 255, 255), 5);
			}
		}
		//background update

		//인스턴스 화면 시각화
		SLAM->VisualizeImage(mapName, vimg, tempVisID);

		//Global Map Update
		if (type == InstanceType::SEG)
		{
			UpdateGaussianObjectMap(mapUpdatedGOs, pPrevSegMask, pCurrSegMask, type);
		}
		if (type == InstanceType::MAP)
		{
			UpdateGaussianObjectMap(mapUpdatedGOs, pReqMask, pCurrSegMask, type);
		}
		//for (auto pair : mapUpdatedGOs)
		//{
		//	int pid = pair.first.first;
		//	int cid = pair.first.second;

		//	auto pG = pair.second;

		//	if (!pG)
		//	{
		//		//이게 발생하는게 말이 안되는데???
		//		continue;
		//	}

		//	if (type == InstanceType::SEG && !pReqMask->GaussianMaps.Get(pid))
		//	{
		//		auto pIns = pPrevSegInstance[pid];
		//		pPrevSegMask->GaussianMaps.Update(pid, pG);
		//		pG->AddObservation(pPrevSegMask, pIns);
		//	}
		//	if (!pCurrSegMask->GaussianMaps.Get(cid))
		//	{
		//		auto pIns = pCurrSegInstance[cid];
		//		pCurrSegMask->GaussianMaps.Update(cid, pG);
		//		pG->AddObservation(pCurrSegMask, pIns);
		//	}
		//}

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
		std::map<int, std::map<int, AssoMatchRes*>> res;
		CalculateIOU(mapRaftInstance, pCurrSegInstance, res);

		//Error case test 중
		//count
		std::map<int, int> mapCountPrev, mapCountCurr;
		std::map<std::pair<int, int>, float> mapErrCase;
		for (auto pair : res)
		{
			auto pid = pair.first;
			
			for (auto pair2 : pair.second)
			{
				auto cid = pair2.first;
				auto ares = pair2.second;
				if(ares->iou > 0.5){
					mapCountCurr[cid]++;
					mapCountPrev[pid]++;
				}
			}
		}
		for (auto pair : res)
		{
			auto pid = pair.first;
			if (pid == 0)
				continue;
			for (auto pair2 : pair.second)
			{
				auto cid = pair2.first;
				if (cid == 0)
					continue;
				int nCount = mapCountCurr[cid];
				auto ares = pair2.second;
				if (nCount > 1 && ares->iou > 0.5)
				{
					auto keypair = std::make_pair(pid, cid);
					mapErrCase[keypair] = ares->iou;
					//std::cout <<keypair.first<<" "<<keypair.second << " == " << ares->iou <<" "<<(int)pPrevSegInstance[pid]->type << std::endl;
				}
			}
		}
		//Error case test 중
		 
		//iou > th
		std::map<std::pair<int, int>, GOMAP::GaussianObject*> mapUpdatedGO; //pair<pid, cid>
		std::map<int, std::set<FrameInstance*>> mapCurrMatchRes; //cid, previnstance
		std::map<FrameInstance*, std::set<int>> mapMapMatchRes;

		//성공한 것은 뉴마스크에(original mask), 샘으로 추가 비교가 필요한 것은 샘마스크에 기록(라프트 인스턴스가 기록)
		//프레브가 여러 컬과 중복되는 경우는? 일단 확인해보고 처리하기~~~~~~~~~~~~~~~~~~~~~~ 확인 후 삭제
		//컬의 중복이 거의 같은 프레브가 존재하는 경우. 샘으로 잘못 추가된 경우가 대부분일 듯? 샘 확인 후 삭제
		auto pNewMask = new InstanceMask();
		InstanceMask* pSAMMask = nullptr;
		if (pNewBF->mapMasks.Count("reqSAM"))
			pSAMMask = pNewBF->mapMasks.Get("reqSAM");
		else{
			pSAMMask = new InstanceMask();
			pSAMMask->id1 = id;
			pSAMMask->id2 = id2;
		}
		std::set<int> sAlready;
		for (auto pair : res)
		{
			auto pid = pair.first;
			auto pPrevIns = pPrevSegInstance[pid];
			auto pRaftIns = mapRaftInstance[pid];

			if (!pPrevIns)
				std::cout << "asso error seg = " << pid << std::endl;
			if (!pRaftIns)
				std::cout << "asso error raft = " << pid << std::endl;

			auto pPrevGO = pPrevGOs[pid];

			for (auto pair2 : pair.second)
			{
				
				auto cid = pair2.first;
				auto ares = pair2.second;

				if (ares->iou > 0.5)
				{
					//if (sAlready.count(pid))
					//{
					//	//error check
					//	std::cout << "Association error = already inserted = " << pid << std::endl;
					//	break;
					//}
					//sAlready.insert(pid);

					if (cid == 0)
					{
						ares->req = true;
						ares->res = false;
						pSAMMask->FrameInstances.Update(pid, pRaftIns);
						pSAMMask->GaussianMaps.Update(pid, pPrevGO);
						pSAMMask->mapResAssociation[pid] = ares;
					}
					else {
						ares->res = true;
						ares->req = false;
						pNewMask->FrameInstances.Update(pid, pPrevIns);
						pNewMask->GaussianMaps.Update(pid, pPrevGO);
						pNewMask->mapResAssociation[pid] = ares;

						mapUpdatedGO[std::make_pair(pid, cid)] = nullptr;
					}
					mapMapMatchRes[pPrevIns].insert(id);
					mapCurrMatchRes[cid].insert(pPrevIns);
				}
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
		pNewBF->mapMasks.Update("reqSAM", pSAMMask);

		//request SAM
		cv::Mat ptdata(0, 1, CV_32FC1);
		auto pSAMIns = pSAMMask->FrameInstances.Get();
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
		UpdateGaussianObjectMap(mapUpdatedGO, pNewMask, pCurrSegMask, InstanceType::SEG);
		//new mask에 객체 추가.
		

		cv::Mat vimg = pNewBF->img.clone();
		VisualizeInstance(mapRaftInstance, pCurrSegInstance, vimg);
		//SLAM->VisualizeImage(mapName, vimg, 0);

		int nGO = 0;
		//for (auto pair : mapUpdatedGO)
		//{
		//	int pid = pair.first.first;
		//	int cid = pair.first.second;

		//	auto pG = pair.second;
		//	if (!pG)
		//	{
		//		//????????이게 말이 되나?
		//		continue;
		//	}
		//	if (!pNewMask->GaussianMaps.Get(pid))
		//	{
		//		auto pIns = pPrevSegInstance[pid];
		//		pPrevSegMask->GaussianMaps.Update(pid, pG);
		//		pG->AddObservation(pPrevSegMask, pIns);
		//		nGO++;
		//	}
		//	if (!pCurrSegMask->GaussianMaps.Get(cid))
		//	{
		//		auto pIns = pCurrSegInstance[cid];
		//		pCurrSegMask->GaussianMaps.Update(cid, pG);
		//		pG->AddObservation(pCurrSegMask, pIns);
		//	}
		//}

		//visualization
		//VisualizeAssociation(SLAM, pNewBF, pPrevBF, pCurrSegMask, mapCurrMatchRes, mapName);
		//VisualizeErrorAssociation(SLAM, pNewBF, pPrevBF, pCurrSegMask, pPrevSegMask, mapErrCase, mapName, 6);

		//std::cout << "Association::test = " <<id<<"=="<< nGO << " == " << mapRaftInstance.size() << " == " << mapMapMatchRes.size() << " " << " ||" << pCurrSegInstance.size() << std::endl;
	}
	void AssociationManager::AssociationWithPrev(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key, const int id
		, const std::string mapName, const std::string userName, BoxFrame* pNewBF, BoxFrame* pPrevBF
		, InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, bool bShow) 
	{
		auto mapCurrSegInstance = pCurrSegMask->FrameInstances.Get();
		auto mapPrevSegInstance = pPrevSegMask->FrameInstances.Get();
		auto tempGOs = pPrevSegMask->GaussianMaps.Get();

		std::map<int, GOMAP::GaussianObject*> mapPrevGOs;
		for (auto pair : tempGOs)
		{
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
			
			pCurrMap2D.GetRect(chi);
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

			for (int i = 0; i < vecMatches.size(); i += 10)
			{
				auto pair2 = vecMatches[i];
				auto pt1 = vecPrev[pair2.first];
				auto pt2 = vecCurr[pair2.second];
				vecPairVisualizedMatches.push_back(std::make_pair(pt1, pt2));
			}
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

		//맵을 현재 프레임에 프로젝션한 것, 현재 프레임의 인스턴스
		std::map<int, std::map<int, AssoMatchRes*>> res; //pid, map<cid, iou>
		CalculateIOU(mapCurrMapInstance, mapCurrSegInstance, res);

		//매칭 연결.
		//다대다 매칭 문제 해결 필요

		//업데이트

		////시각화
		////test visualization
		//cv::Mat vimg = pNewBF->img.clone();
		//for (auto pair : mapCurrMapInstance)
		//{
		//	auto pGO = mapPrevGOs[pair.first];
		//	auto pIns = pair.second;
		//	cv::rectangle(vimg, pIns->rect, cv::Scalar(255, 0, 0), 2);
		//	cv::putText(vimg, std::to_string(pGO->id), pIns->pt, 2, 1.3, cv::Scalar(0, 0, 255), 2);
		//}
		//for (auto pair : mapCurrSegInstance)
		//{
		//	auto pIns = pair.second;
		//	cv::rectangle(vimg, pIns->rect, cv::Scalar(255, 255, 0), 2);
		//}
		//SLAM->VisualizeImage(mapName, vimg, 5);
		//test visualization

		//여기서 실패한 객체에 대해서 다음 로컬 맵에서 매칭하도록 해야 함.
	}
	void AssociationManager::AssociationWithMAP(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key, const int id
		, const std::string mapName, const std::string userName, BoxFrame* pNewBF, BoxFrame* pPrevBF
		, InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, bool bShow) {

		//prev 정보를 내부에서 얻도록 변경.
		auto pCurrKF = pNewBF->mpRefKF;
		auto pPrevKF = pPrevBF->mpRefKF;
		auto pCurrSegInstance = pCurrSegMask->FrameInstances.Get();

		std::map<int, GOMAP::GaussianObject*> mapLocalGOs;
		std::map<int, GOMAP::GO2D> mapLocalPrevGO2Ds, mapLocalCurrGO2Ds;
		std::map<int, FrameInstance*> mapMapPrevFrameInstance, mapMapCurrFrameInstance;
		GetLocalObjectMaps(pPrevSegMask, mapLocalGOs);
		//convert prev frame
		GetObjectMap2Ds(mapLocalGOs, pPrevBF, mapLocalPrevGO2Ds);
		ConvertFrameInstances(mapLocalPrevGO2Ds, pPrevBF, mapMapPrevFrameInstance);
		//convert curr frame
		GetObjectMap2Ds(mapLocalGOs, pNewBF, mapLocalCurrGO2Ds);
		ConvertFrameInstances(mapLocalCurrGO2Ds, pNewBF, mapMapCurrFrameInstance);

		//매칭 정보
		std::map<int, std::vector<cv::Point2f>> mapPrevKPs, mapCurrKPs;
		std::map<int, cv::Mat> mapPrevDescs, mapCurrDescs;

		//test visualization
		float chi = sqrt(5.991);
		cv::Mat vimg = pNewBF->img.clone();
		

		std::vector<cv::Point2f> mvPrevKeys, mvCurrKeys;
		cv::Mat prevDesc = cv::Mat::zeros(0, 32, CV_8UC1);
		cv::Mat currDesc = cv::Mat::zeros(0, 32, CV_8UC1);

		for (auto pair : mapMapCurrFrameInstance)
		{
			auto pGO = mapLocalGOs[pair.first];
			auto pIns = pair.second;
			cv::rectangle(vimg, pIns->rect, cv::Scalar(255, 0, 0), 2);

			auto g = mapLocalPrevGO2Ds[pair.first];
			cv::Size newSize(cvRound(g.major * chi), cvRound(g.minor * chi));
			cv::ellipse(vimg, pIns->pt, newSize, g.angle_deg, 0, 360, cv::Scalar(255, 255, 0), 3);
			cv::putText(vimg, std::to_string(pGO->id), pIns->pt, 2, 1.3, cv::Scalar(0, 0, 255), 2);

			//현재 프레임의 포인트 후보군 획득

			//이전 프레임에서 

			////피쳐 출력
			//g.GetRect(chi);
			//auto rect = g.rect;
			//cv::ellipse(vimg2, pIns->pt, newSize, g.angle_deg, 0, 360, cv::Scalar(255, 255, 0), 3);
			//for(int i = 0; i < pKF->mvKeys.size(); i++)
			//{
			//	auto kp = pKF->mvKeys[i];
			//	if (rect.contains(kp.pt))
			//	{
			//		mvPrevKeys.push_back(kp.pt);
			//		prevDesc.push_back(pKF->mDescriptors.row(i));
			//		cv::circle(vimg2, kp.pt, 5, cv::Scalar(255, 0, 0), -1);
			//	}
			//}
		}
		for (auto pair : pCurrSegInstance)
		{
			auto pIns = pair.second;
			cv::rectangle(vimg, pIns->rect, cv::Scalar(255, 255, 0), 2);

			/*cv::rectangle(vimg2, pIns->rect, cv::Scalar(0, 0, 255), 2);
			for (int i = 0; i < pKF->mvKeys.size(); i++)
			{
				auto kp = pKF->mvKeys[i];
				if (pIns->rect.contains(kp.pt))
				{
					mvCurrKeys.push_back(kp.pt);
					currDesc.push_back(pKF->mDescriptors.row(i));
					cv::circle(vimg2, kp.pt, 3, cv::Scalar(0, 0, 255), -1);
				}
			}*/
		}

		//전체 매칭 테스트

		//전체 매칭 테스트

		SLAM->VisualizeImage(mapName, vimg, 5);
		//SLAM->VisualizeImage(mapName, vimg2, 1);
		//test visualization

		////수정 버전
		//std::map<GOMAP::GaussianObject*, FrameInstance*> mapInstance; //type은 MAP
		//ProjectObjectMap(pCurrSegMask, pPrevSegMask, pNewBF, mapInstance);

		//std::map<int, GOMAP::GaussianObject*> mapMap3D;
		//std::map<int, FrameInstance*> mapMap2D;
		//for (auto pair : mapInstance)
		//{
		//	int id = pair.first->id;
		//	mapMap2D[id] = pair.second;
		//	mapMap3D[id] = pair.first;
		//}
		////수정 버전

		//MAP MASK 생성
		//id는 GO의 global id임.
		auto pMapMask = new InstanceMask();
		auto pSamMask = new InstanceMask();

		//iou matching
		std::map<int, std::map<int, AssoMatchRes*>> res; //pid, map<cid, iou>
		CalculateIOU(mapMapCurrFrameInstance, pCurrSegInstance, res);

		//iou > th
		//시각화 코드는 수정이 필요할 듯
		std::map<std::pair<int, int>, GOMAP::GaussianObject*> mapUpdatedGO; //pair<pid, cid>
		std::map<int, std::set<FrameInstance*>> mapCurrMatchRes;
		//std::map<FrameInstance*, std::set<int>> mapMapMatchRes;
		for (auto pair : res)
		{
			auto mid = pair.first;
			auto pPrevIns = mapMapPrevFrameInstance[mid];
			auto pCurrIns = mapMapCurrFrameInstance[mid];
			auto pGO = mapLocalGOs[mid];

			//중복 문제에 대한 해결이 필요함.
			for (auto pair2 : pair.second)
			{
				auto cid = pair2.first;
				auto ares = pair2.second;

				if (ares->iou > 0.5)
				{

					if (cid == 0)
					{
						//req SAM
						ares->req = true;
						ares->res = false;

						pSamMask->FrameInstances.Update(mid, pCurrIns);
						pSamMask->GaussianMaps.Update(mid, pGO);
					}
					else {
						//matching
						ares->res = true;
						ares->req = false;

						pMapMask->FrameInstances.Update(mid, pPrevIns);
						pMapMask->GaussianMaps.Update(mid, pGO);

						mapUpdatedGO[std::make_pair(mid, cid)] = nullptr;
					}

					//mapMapMatchRes[pF].insert(cid);
					mapCurrMatchRes[cid].insert(pPrevIns);

				}
			}
		}

		pNewBF->mapMasks.Update("ObjMap", pMapMask);
		pNewBF->mapMasks.Update("reqSAM_MAP", pSamMask);

		//request SAM
		cv::Mat ptdata(0, 1, CV_32FC1);
		auto pSAMIns = pSamMask->FrameInstances.Get();
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
			
			std::string tsrc = userName + ".Image." + std::to_string(nobj) + "." + std::to_string(pPrevBF->mnId) + "." + std::to_string((int)InstanceType::MAP);
			auto sam2key = "reqsam2";
			std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();
			auto du_upload = Utils::SendData(sam2key, tsrc, ptdata, id, 15, t_start.time_since_epoch().count());
		}

		UpdateGaussianObjectMap(mapUpdatedGO, pMapMask, pCurrSegMask, InstanceType::MAP);
		//VisualizeAssociation(SLAM, pNewBF, pPrevBF, pCurrSegMask, mapCurrMatchRes, mapName, 6);
		return;
		

		

		//update
		//std::cout << "AssociationWithMAP::test = " << mapMap2D.size() << " " << mapMapMatchRes.size() << std::endl;

	}
	
}
