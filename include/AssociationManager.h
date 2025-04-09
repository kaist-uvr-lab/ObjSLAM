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
		//���� ������ ��
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

		//iou üũ
		static float CalculateMeasureForAsso(const cv::Mat& mask1, const cv::Mat& mask2, float area1, float area2, ObjectMeasureType& type, int id = 1);
		static float CalculateIOU(const cv::Mat& mask1, const cv::Mat& mask2, float area1, int id = 1);
		static float CalculateIOU(const float& i, const float& u);
		static ConcurrentMap<int, int> DebugAssoSeg, DebugAssoSAM;
	public:
		//����� ȯ�� ����
		static bool mbSAM, mbObjBaselineTest;
	private:
		//����Ʈ ������ �̿��ؼ� source �������� ����ũ�� ��ȯ
		static void ConvertMaskWithRAFT(const std::map<int, FrameInstance*>& mapSourceInstance, std::map<int, FrameInstance*>& mapRaftInstance, EdgeSLAM::KeyFrame* pRefKF, const cv::Mat& flow);
		 
		//���ο� ��ҽÿ��̼�. ��� ��ҽÿ��̼��� ������. 
		static void CalculateIOU(const std::map<int, FrameInstance*>& mapPrevInstance, const std::map<int, FrameInstance*>& mapCurrInstance
			, std::map<std::pair<int,int>, AssoMatchRes*>& mapIOU, float th = 0.5);
		
		static void EvaluateMatchResults(std::map<std::pair<int,int>, AssoMatchRes*>& mapIOU, std::map<std::pair<int,int>, AssoMatchRes*>& mapSuccess, float th = 0.5);

		//���� ��Ȯ�̼��� ���� ����
		static void TestUncertainty(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, AssoFramePairData* pPairData);
		static void AssociationWithUncertainty(EdgeSLAM::SLAM* SLAM, AssoFramePairData* pPairData);
		static void AssociationPrevMapWithUncertainty(EdgeSLAM::SLAM* SLAM, AssoFramePairData* pPairData);
		static void AssociateLocalMapWithUncertainty(EdgeSLAM::SLAM* SLAM, AssoFramePairData* pPairData);
		static FrameInstance* GenerateFrameInsWithUncertainty(
			EdgeSLAM::KeyFrame* pKF, const ObjectRegionFeatures& prev, const ObjectRegionFeatures& map, 
			const std::vector<std::pair<int, int>>& vecMatches);
		//ClassifyMatchResults

		////�� ����ũ ��ҽÿ��̼�. ������ Ȯ���� �ʿ��� ������ ����. ������ �ʿ�
		//static int AssociationWithMask(const std::map<int, FrameInstance*>& mapMaskInstance, const std::map<int, FrameInstance*>& mapSourceInstance
		//	, std::vector<AssoMatchRes*>& mapAssoRes, float th = 0.5);
		////������Ʈ ���� �����ӿ� ��������
		//
		////����ũ�� ��ü �� ��ҽÿ��̼�
		//static void AssociationWithMap();
		////���� ���׸����̼� ��ҽÿ��̼�
		//static void AssociationWithSAM(InstanceMask* pCurrSeg, const std::map<int, FrameInstance*>& mapSamInstance, std::map<int, FrameInstance*>& mapMissingInstance, std::vector<AssoMatchRes*>& vecResAsso, float th = 0.5);
		
		static void ProjectObjectMap(InstanceMask* pCurrSeg, InstanceMask* pPrevSeg,
			BoxFrame* pNewBF,
			std::map<GOMAP::GaussianObject*, FrameInstance*>& mapInstance);
		//�� ��û
		static void RequestSAM(BoxFrame* pNewBF, std::map<int, FrameInstance*>& mapMaskInstance, const std::map<GlobalInstance*, cv::Rect>& mapGlobalRect, const std::vector<AssoMatchRes*>& vecResAsso, int id1, int id2, const std::string& userName);
		//����þ� ��ü ����
		static void UpdateGaussianObjectMap2(std::map<int, int>& mapRes, InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, const InstanceType& type);
		static void UpdateGaussianObjectMap(std::map<int,int>& mapRes, InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, const InstanceType& type);
		static void UpdateBaselineObjectMap(std::map<int, int>& mapRes, InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, const InstanceType& type, bool bTest = false);
		//�ð�ȭ
		//��ġ�� �ν��Ͻ�, �� ����� �и�
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

		//�� �ν��Ͻ��� ���� �����ӿ� �߰�. ����� �� �ν��Ͻ� ���� Ȯ���ϴ� ����
		static bool CheckAddNewInstance(std::map<int, FrameInstance*>& mapInstatnces, FrameInstance* pNew);
		static int AddNewInstance(InstanceMask* pFrame, FrameInstance* pNew);

		//rect�� �簢���� �������� ��ȯ
		static void ConvertContour(const cv::Rect& rect, std::vector<cv::Point>& contour);
		//�� rect�� ��ġ���� Ȯ��.//rect2�� rect1�� ���ԵǴ��� Ȯ��
		static bool CheckOverlap(const cv::Rect& rect, const std::vector<cv::Point>& contour);
		static bool CheckOverlap(const cv::Rect& rect1, const cv::Rect& rect2);
		static bool IsContain(const cv::Rect& rect1, const cv::Rect& rect2);
		//KP or MP ��Ī ����
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