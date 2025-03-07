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
	class KeyFrame;
}

namespace ObjectSLAM {

	class ObjectSLAM;
	class BoxFrame;
	class InstanceMask;
	class FrameInstance;
	class GlobalInstance;
	class AssoMatchRes;
	
	namespace GOMAP {
		class GaussianObject;
		class GO2D;
	}

	class AssociationManager {
	public:
		
		static void Association(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key, const int id, const int id2
			, const std::string& mapName, const std::string& userName
			, BoxFrame* pNewBF, BoxFrame* pPrevBF, InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, InstanceMask* pRaft, bool bShow = true);
		static void AssociationWithPrev(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key, const int id
			, const std::string mapName, const std::string userName
			, BoxFrame* pNewBF, BoxFrame* pPrevBF, InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, bool bShow = true);
		static void AssociationWithMAP(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key, const int id
			, const std::string mapName, const std::string userName
			, BoxFrame* pNewBF, BoxFrame* pPrevBF, InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, bool bShow = true);
		static void AssociationWithSAM(EdgeSLAM::SLAM* SLAM, ObjectSLAM* ObjSLAM, const std::string& key
			, const int id, const int id2, const int _type
			, const std::string& mapName, const std::string& userName
			, BoxFrame* pNewBF, InstanceMask* pCurrSegMask
			, std::map<int, FrameInstance*>& mapSamInstances, bool bShow = true);
		
	private:
		//����Ʈ ������ �̿��ؼ� source �������� ����ũ�� ��ȯ
		static void ConvertMaskWithRAFT(const std::map<int, FrameInstance*>& mapSourceInstance, std::map<int, FrameInstance*>& mapRaftInstance, EdgeSLAM::KeyFrame* pRefKF, const cv::Mat& flow);
		 
		//���ο� ��ҽÿ��̼�. ��� ��ҽÿ��̼��� ������. 
		static void CalculateIOU(const std::map<int, FrameInstance*>& mapPrevInstance, const std::map<int, FrameInstance*>& mapCurrInstance
			, std::map<int, std::map<int, AssoMatchRes*>>& mapIOU, float th = 0.5);
		
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
		static void UpdateGaussianObjectMap(std::map<std::pair<int, int>, GOMAP::GaussianObject*>& mapGO, InstanceMask* pPrevSegMask, InstanceMask* pCurrSegMask, const InstanceType& type);
		
		//�ð�ȭ
		//��ġ�� �ν��Ͻ�, �� ����� �и�
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

		////LocalMaps
		static void GetLocalObjectMaps(InstanceMask* pMask, std::map<int,GOMAP::GaussianObject*>& spGOs);
		////Project
		static void GetObjectMap2Ds(const std::map<int, GOMAP::GaussianObject*>& spGOs, BoxFrame* pBF, std::map<int, GOMAP::GO2D>& spG2Ds);
		static void ConvertFrameInstances(const std::map<int, GOMAP::GO2D>& spG2Ds, BoxFrame* pBF, std::map<int, FrameInstance*>& mapIns);
	};
}
#endif