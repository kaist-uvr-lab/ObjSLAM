#ifndef OBJECT_SLAM_INSTANCE_LINKER_H
#define OBJECT_SLAM_INSTANCE_LINKER_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

#include <ObjectSLAM.h>

#include <ObjectSLAMTypes.h>
#include <HungarianMatcher.h>

namespace EdgeSLAM {
    class KeyFrame;
    class Frame;
}

namespace ObjectSLAM {

	class SegInstance;
    class BoxFrame;
    class Instance;
    class GlobalInstance;
    class ObjectSLAM;

    class InstanceSim {
    public:

        static bool CheckStaticObject(const std::vector<cv::Point>& contour, std::map<int, Instance*>& mapInstances, int th = 3);

        static void FindOverlapMP(Instance* a, Instance* b, std::set<EdgeSLAM::MapPoint*>& setMPs);
        static void FindOverlapMP(Instance* a, EdgeSLAM::Frame* pF, std::set<EdgeSLAM::MapPoint*>& setMPs);    
        static void FindOverlapMP(Instance* a, EdgeSLAM::KeyFrame* pKF, std::set<EdgeSLAM::MapPoint*>& setMPs);

        static float ComputeSimFromMP(Instance* a, Instance* b);
        static float ComputeSimFromPartialMP(Instance* a, Instance* b);
        static float ComputeSimFromPartialMP(Instance* a, EdgeSLAM::Frame* b);
        static float ComputeSimFromPartialMP(Instance* a, EdgeSLAM::KeyFrame* b);
        static float ComputeSimFromIOU(Instance* a, Instance* b);
        static float ComputeSimFromIOU(const cv::Mat& mask1, const cv::Mat& mask2);


        static bool ComputSim(const std::vector<cv::Point>& contours, const std::vector<cv::Point2f>& pts, float& val, float th = 0.5);
        
    };

    class InstanceLinker {
    public:
        InstanceLinker(){}
        virtual ~InstanceLinker(){}
    public:

        static ObjectSLAM* ObjectSystem;
        static void SetSystem(ObjectSLAM* p);

        static void FindInstances(BoxFrame* curr, BoxFrame* prev, std::set<int>& setMatchIDs, std::vector<cv::Point2f>& vecPoints, bool bShow = false);

        static void LinkNeighBFs(BoxFrame* curr, std::set<int>& setMatchIDs, bool bShow = false);

        //현재 프레임과 이전 프레임에 대해서 매칭함.
        static void computeSim(BoxFrame* prev, BoxFrame* curr, const std::vector<std::pair<int, int>>& vecPairMatches, float iou_threshold = 0.5);

        static void compute(BoxFrame* prev, BoxFrame* curr, float iou_threshold = 0.5);
        static void computeFromOF(BoxFrame* prev, BoxFrame* curr,
            const std::vector<std::pair<int,int>>& vecPairMatchInstance, 
            std::map<int,int>& assignments, std::map<std::pair<int,int>, std::pair<int,int>>& mapChanged, float iou_threshold = 0.5);
        static void computeFromMP(BoxFrame* prev, BoxFrame* curr,
            const std::vector<std::pair<int, int>>& vecPairMatchInstance,
             std::map<int, int>& assignments, std::map<std::pair<int, int>, std::pair<int, int>>& mapChanged, float iou_threshold = 0.5);
        static std::vector<std::pair<int,int>> computeFromOF(BoxFrame* prev, BoxFrame* curr, 
            const std::vector<cv::Point2f>& vecPrevCorners, const std::vector<cv::Point2f>& vecCurrCorners, const std::vector<int>& vecMatchingIDXs, float iou_threshold = 0.5);

        static std::map<int, std::vector<std::pair<int, int>>> linkInstancesAcrossKeyframes(
            const std::vector<SegInstance*>& vecInstances, float iou_threshold = 0.5) {
            std::map<int, std::vector<std::pair<int, int>>> global_instances;
            int next_global_id = 0;

            //for (int frame_idx = 0; frame_idx < keyframes.size(); ++frame_idx) {
            //    const auto& instances = keyframes[frame_idx].instances;

            //    if (frame_idx == 0) {
            //        // For the first frame, assign new global IDs to all instances
            //        for (int i = 0; i < instances.size(); ++i) {
            //            global_instances[next_global_id] = { {frame_idx, i} };
            //            ++next_global_id;
            //        }
            //    }
            //    else {
            //        const auto& prev_instances = keyframes[frame_idx - 1].instances;
            //        cv::Mat iou_matrix = computeIoUMatrix(instances, prev_instances);
            //        std::vector<int> assignments = hungarianMatching(iou_matrix);

            //        for (int i = 0; i < assignments.size(); ++i) {
            //            int j = assignments[i];
            //            if (j != -1 && iou_matrix.at<float>(i, j) > iou_threshold) {
            //                // Find the global ID for the matched previous instance
            //                for (auto& [global_id, instance_list] : global_instances) {
            //                    if (!instance_list.empty() &&
            //                        instance_list.back().first == frame_idx - 1 &&
            //                        instance_list.back().second == j) {
            //                        instance_list.push_back({ frame_idx, i });
            //                        break;
            //                    }
            //                }
            //            }
            //            else {
            //                // If no match found, create a new global instance
            //                global_instances[next_global_id] = { {frame_idx, i} };
            //                ++next_global_id;
            //            }
            //        }
            //    }
            //}

            return global_instances;
        }

    private:
        //map point
        //겹치는 맵포인트
        //key point
        //둘의 매칭 정확도
        static cv::Mat computeIoUMatrixWithMPs(
            const std::map<int, SegInstance*>& instances1,
            const std::map<int, SegInstance*>& instances2);

        static cv::Mat computeIoUMatrixWithKPs(
            const std::map<int, SegInstance*>& instances1,
            const std::map<int, SegInstance*>& instances2);

        //iou 계산
        static cv::Mat computeIoUMatrix(const std::vector<SegInstance*>& instances1,
            const std::vector<SegInstance*>& instances2) {
            cv::Mat iou_matrix(instances1.size(), instances2.size(), CV_32F);

            for (int i = 0; i < instances1.size(); ++i) {
                for (int j = 0; j < instances2.size(); ++j) {
                    cv::Mat intersection, union_;
                    //cv::bitwise_and(instances1[i].mask, instances2[j].mask, intersection);
                    //cv::bitwise_or(instances1[i].mask, instances2[j].mask, union_);

                    float iou = static_cast<float>(cv::countNonZero(intersection)) /
                        static_cast<float>(cv::countNonZero(union_));
                    iou_matrix.at<float>(i, j) = iou; 
                }
            }

            return iou_matrix;
        }

        static void hungarianMatching(const cv::Mat& cost_matrix, std::vector<int>& assignment);
    };
}


#endif