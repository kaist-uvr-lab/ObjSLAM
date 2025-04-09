#ifndef OBJECT_SLAM_ASSOCIATION_MANAGER_H
#define OBJECT_SLAM_ASSOCIATION_MANAGER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>
#include <ObjectSLAMTypes.h>

namespace EdgeSLAM {
	class SLAM;
	class Frame;
	class KeyFrame;
	class MapPoint;
}

namespace ObjectSLAM {

	namespace Evaluation {
		class EvalObj;
	}

	class ObjectSLAM;
	class BoxFrame;
	class InstanceMask;
	class FrameInstance;
	class GlobalInstance;
	class AssoMatchRes;
	class AssoFramePairData;
	class ObjectRegionFeatures;

	namespace GOMAP {
		class GaussianObject;
		class GO2D;
	}

	class AssociationManager {
	public:
		//최종 마무리 전
		static void AssociationWithSeg(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key, const std::string& mapName, const std::string& userName, AssoFramePairData* pPairData);

		static void AssociationWithSAM(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key
			, const std::string& mapName, const std::string& userName, const int _type
			, AssoFramePairData* pPairData, InstanceMask* pMask);

		static void AssociationWithSAMfromSEG(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key
			, const std::string& mapName, const std::string& userName, const InstanceType& type
			, AssoFramePairData* pPairData);
		static void AssociationWithSAMfromMAP(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key
			, const std::string& mapName, const std::string& userName, const InstanceType& type
			, AssoFramePairData* pPairData);

		/*
		EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key
			, const std::string& mapName, const std::string& userName, const int _type
			, BoxFrame* pCurr, EdgeSLAM::Frame* pFrame
		*/
		static bool AssociationWithFrame(BoxFrame* pCurr, EdgeSLAM::Frame* pFrame, const cv::Mat& K, const cv::Mat& T,
			std::map<GOMAP::GaussianObject*, FrameInstance*>& mapRes);

		//iou 체크
		static float CalculateMeasureForAsso(const cv::Mat& mask1, const cv::Mat& mask2, float area1, float area2, ObjectMeasureType& type, int id = 1);
		static float CalculateIOU(const cv::Mat& mask1, const cv::Mat& mask2, float area1, int id = 1);
		static float CalculateIOU(const float& i, const float& u);
		static ConcurrentMap<int, int> DebugAssoSeg, DebugAssoSAM;
	public:
		//실험용 환경 설정
		static bool mbSAM, mbObjBaselineTest;
	private:
		//라프트 정보를 이용해서 source 프레임의 마스크를 변환
		static void ConvertMaskWithRAFT(const std::map<int, FrameInstance*>& mapSourceInstance, std::map<int, FrameInstance*>& mapRaftInstance, EdgeSLAM::KeyFrame* pRefKF, const cv::Mat& flow);
		 
		//새로운 어소시에이션. 모든 어소시에이션을 수행함. 
		static void CalculateIOU(const std::map<int, FrameInstance*>& mapPrevInstance, const std::map<int, FrameInstance*>& mapCurrInstance
			, std::map<std::pair<int,int>, AssoMatchRes*>& mapIOU, float th = 0.5);
		
		static void EvaluateMatchResults(std::map<std::pair<int,int>, AssoMatchRes*>& mapIOU, std::map<std::pair<int,int>, AssoMatchRes*>& mapSuccess, float th = 0.5);

		//맵의 불확싱성을 위한 연결
		static void TestUncertainty(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, AssoFramePairData* pPairData);
		static void AssociationWithUncertainty(EdgeSLAM::SLAM* SLAM, AssoFramePairData* pPairData);
		static void AssociationPrevMapWithUncertainty(EdgeSLAM::SLAM* SLAM, AssoFramePairData* pPairData);
		static void AssociateLocalMapWithUncertainty(EdgeSLAM::SLAM* SLAM, AssoFramePairData* pPairData);
		static FrameInstance* GenerateFrameInsWithUncertainty(
			EdgeSLAM::KeyFrame* pKF, const ObjectRegionFeatures& prev, const ObjectRegionFeatures& map, 
			const std::vector<std::pair<int, int>>& vecMatches);
		//ClassifyMatchResults

		////두 마스크 어소시에이션. 샘으로 확인이 필요한 개수를 리턴. 수정이 필요
		//static int AssociationWithMask(const std::map<int, FrameInstance*>& mapMaskInstance, const std::map<int, FrameInstance*>& mapSourceInstance
		//	, std::vector<AssoMatchRes*>& mapAssoRes, float th = 0.5);
		////오브젝트 맵을 프레임에 프로젝션
		//
		////마스크와 객체 맵 어소시에이션
		//static void AssociationWithMap();
		////샘과 세그멘테이션 어소시에이션
		//static void AssociationWithSAM(InstanceMask* pCurrSeg, const std::map<int, FrameInstance*>& mapSamInstance, std::map<int, FrameInstance*>& mapMissingInstance, std::vector<AssoMatchRes*>& vecResAsso, float th = 0.5);
		
		static void ProjectObjectMap(InstanceMask* pCurrSeg, InstanceMask* pPrevSeg,
			BoxFrame* pNewBF,
			std::map<GOMAP::GaussianObject*, FrameInstance*>& mapInstance);
		//샘 요청
		static void RequestSAM(BoxFrame* pNewBF, std::map<int, FrameInstance*>& mapMaskInstance, const std::map<GlobalInstance*, cv::Rect>& mapGlobalRect, const std::vector<AssoMatchRes*>& vecResAsso, int id1, int id2, const std::string& userName);
		//가우시안 객체 관리
		static void UpdateGaussianObjectMap2(std::map<int, int>& mapRes, InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, const InstanceType& type);
		static void UpdateGaussianObjectMap(std::map<int,int>& mapRes, InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, const InstanceType& type);
		static void UpdateBaselineObjectMap(std::map<int, int>& mapRes, InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, const InstanceType& type, bool bTest = false);
		//시각화
		//매치와 인스턴스, 맵 출력을 분리
		static void VisualizeAssociation(EdgeSLAM::SLAM* SLAM, AssoFramePairData* pPairData, std::string mapName, int num_vis = 0, int type = 0);
		
		static void VisualizeInstance(const std::map<int, GOMAP::GaussianObject*>& pMAP, std::map<int, FrameInstance*>& pPrev, const std::map<int, FrameInstance*>& pCurr, cv::Mat& vimg);
		static void VisualizeInstance(const std::map<int, FrameInstance*>& pPrev, const std::map<int, FrameInstance*>& pCurr, cv::Mat& vimg);
		static void VisualizeObjectMap(cv::Mat& res, InstanceMask* pMask, BoxFrame* pBF);

		static void VisualizeMatchAssociation(EdgeSLAM::SLAM* SLAM, BoxFrame* pNewBF, BoxFrame* pPrevBF
			, InstanceMask* pCurrSeg, std::map<int, std::set<FrameInstance*>>& mapRes, std::string mapName, int num_vis = 0);
		static void VisualizeAssociation(EdgeSLAM::SLAM* SLAM, BoxFrame* pNewBF, BoxFrame* pPrevBF
			, InstanceMask* pCurrSeg, InstanceMask* pPrevSeg, std::string mapName, int num_vis = 0);
		static void VisualizeAssociation(EdgeSLAM::SLAM* SLAM, BoxFrame* pNewBF, BoxFrame* pPrevBF
			, InstanceMask* pCurrSeg, std::map<int, std::set<FrameInstance*>>& mapRes, std::string mapName, int num_vis = 0);
		static void VisualizeErrorAssociation(EdgeSLAM::SLAM* SLAM, BoxFrame* pNewBF, BoxFrame* pPrevBF
			, InstanceMask* pCurrSeg, InstanceMask* pPrevSeg, std::map<std::pair<int, int>, float>& mapErrCase, std::string mapName, int num_vis = 0);

		//샘 인스턴스를 현재 프레임에 추가. 젣디로 된 인스턴스 인지 확인하는 과정
		static bool CheckAddNewInstance(std::map<int, FrameInstance*>& mapInstatnces, FrameInstance* pNew);
		static int AddNewInstance(InstanceMask* pFrame, FrameInstance* pNew);

		//rect의 사각형을 윤곽으로 변환
		static void ConvertContour(const cv::Rect& rect, std::vector<cv::Point>& contour);
		//두 rect가 겹치는지 확인.//rect2가 rect1에 포함되는지 확인
		static bool CheckOverlap(const cv::Rect& rect, const std::vector<cv::Point>& contour);
		static bool CheckOverlap(const cv::Rect& rect1, const cv::Rect& rect2);
		static bool IsContain(const cv::Rect& rect1, const cv::Rect& rect2);
		//KP or MP 매칭 관련
		static void ExtractRegionFeatures(EdgeSLAM::Frame* pF, const cv::RotatedRect& rect, std::vector<cv::KeyPoint>& vecKPs, cv::Mat& desc);
		static void ExtractRegionFeatures(EdgeSLAM::Frame* pF, const cv::RotatedRect& rect, std::vector<cv::KeyPoint>& vecKPs, std::vector<EdgeSLAM::MapPoint*>& vecMPs, cv::Mat& desc);
		static void ExtractRegionFeatures(EdgeSLAM::KeyFrame* pKF, const cv::RotatedRect& rect, std::vector<cv::KeyPoint>& vecKPs, cv::Mat& desc);
		static void ExtractRegionFeatures(EdgeSLAM::KeyFrame* pKF, const cv::RotatedRect& rect, std::vector<cv::KeyPoint>& vecKPs, std::vector<EdgeSLAM::MapPoint*>& vecMPs, cv::Mat& desc);
		static void ExtractRegionFeatures(EdgeSLAM::KeyFrame* pKF, const cv::Rect& rect, std::vector<cv::KeyPoint>& vecKPs, cv::Mat& desc);
		static void ExtractRegionFeatures(EdgeSLAM::KeyFrame* pKF, const cv::Rect& rect, std::vector<cv::KeyPoint>& vecKPs, std::vector<EdgeSLAM::MapPoint*>& vecMPs, cv::Mat& desc);

		////LocalMaps
		static void GetLocalObjectMaps(InstanceMask* pMask, std::map<int,GOMAP::GaussianObject*>& spGOs);
		////Project
		static void GetObjectMap2Ds(const std::map<int, GOMAP::GaussianObject*>& spGOs, BoxFrame* pBF, std::map<int, GOMAP::GO2D>& spG2Ds);
		static void ConvertFrameInstances(const std::map<int, GOMAP::GO2D>& spG2Ds, BoxFrame* pBF, std::map<int, FrameInstance*>& mapIns);
	
		//request sam
		static void RequestSAMToServer(const std::map<int, FrameInstance*> mapRes, int currid, int previd, const std::string& user, InstanceType type);
};
}
#endif