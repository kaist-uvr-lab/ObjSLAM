#include <InstanceLinker.h>

#include <SegInstance.h>
#include <BoxFrame.h>
#include <KeyFrame.h>
#include <MapPoint.h>
#include <SemanticLabel.h>

namespace ObjectSLAM {

    void InstanceLinker::computeFromMP (BoxFrame* prev, BoxFrame* curr, 
        const std::vector<std::pair<int, int>>& vecPairMatchInstance,
        std::map<int, int>& assignments, std::map<std::pair<int, int>, std::pair<int, int>>& mapChanged, float iou_threshold)
    {
        auto prevKF = prev->mpRefKF;
        auto currKF = curr->mpRefKF;
 
        //mp 집합
        std::vector<std::pair<int, int>> avecPairMatchInstance;
        for (auto pair : prev->mmpBBs) {
            int sid = pair.first;
            auto pInstance = pair.second;

            for (int i = 0; i < pInstance->mvpMapPoints.size(); i++) {
                auto pMPi = pInstance->mvpMapPoints[i];

                if (!pMPi || pMPi->isBad())
                    continue;

                auto kp = pInstance->mvKeyPoints[i];   

                int cid = curr->GetFrameInstanceId(pMPi);
                if (cid >= 0)
                {
                    avecPairMatchInstance.push_back(std::make_pair(sid, cid));
                }

                /*int idx = pMPi->GetIndexInKeyFrame(currKF);
                if (idx >= 0) {
                    auto currPt = currKF->mvKeys[idx].pt;
                    int cid = curr->seg.at<uchar>(currPt);

                    avecPairMatchInstance.push_back(std::make_pair(sid, cid));
                }*/
             }
        }
        computeFromOF(prev, curr, avecPairMatchInstance, assignments, mapChanged);

        return;
        int maxPrev = prev->mnMaxID;
        int maxCurr = curr->mnMaxID;
        std::map<std::pair<int, int>, int> mapLinkCount;
        std::map<int, int> mapPrevCount, mapCurrCount;

        for (auto pair : avecPairMatchInstance) {
            int pid = pair.first;
            int cid = pair.second;

            /*if (pid > maxPrev)
                maxPrev = pid;
            if (cid > maxCurr)
                maxCurr = cid;*/

            mapPrevCount[pid]++;
            mapCurrCount[cid]++;
            mapLinkCount[std::make_pair(pid, cid)]++;
        }

        std::vector<std::pair<int, int>> res;
        //sstd::cout << "link::2" << std::endl;
        cv::Mat iou_matrix = cv::Mat::zeros(maxPrev + 1, maxCurr + 1, CV_32F);
        for (auto ppair : mapPrevCount) {
            int pid = ppair.first;
            float pcount = (float)ppair.second;

            auto prevIns = prev->mmpBBs[pid];
            int plabel = prevIns->mpConfLabel->label;
            auto pIsthing = prevIns->isObject();
            bool bPrevTable = prevIns->isTable();

            for (auto cpair : mapCurrCount) {
                int cid = cpair.first;
                float ccount = (float)cpair.second;
                
                auto currIns = curr->mmpBBs[cid];
                int clabel = currIns->mpConfLabel->label;
                auto cIsthing = currIns->mbIsthing;
                bool bCurrTable = currIns->isTable();
                
                auto pair = std::make_pair(pid, cid);
                float count = (float)mapLinkCount[pair];

                float sum = pcount + ccount - count;
                float val = count / sum;

                if (count < 5)
                    continue;

                {
                    auto pratio = count / pcount;
                    auto cratio = count / ccount;
                    auto ratio = count / sum;

                    bool bNotInstance = ratio < iou_threshold;
                    
                    bool bMerge = false;
                    std::string strLabel = "";
                    if (bNotInstance && pratio > 0.8 && pIsthing) {
                        //일단 다이닝 테이블인 경우 추가 60
                        if (bCurrTable && !bPrevTable) {
                            //새로운 인스턴스를 추가해야 함.
                            cv::Mat newCol = cv::Mat::zeros(maxPrev + 1, 1, CV_32F);
                            cv::hconcat(iou_matrix, newCol, iou_matrix);

                            cid = ++maxCurr;
                            val = pratio;
                            pair.second = cid;
                            //std::cout << "InstanceLinker::Add New Instance = " << prevIns->mStrLabel<< std::endl;
                            //auto newIns = new SegInstance(curr, curr->fx, curr->fy, curr->cx, curr->cy, plabel, prevIns->mfConfidence, prevIns->mbIsthing, nullptr);
                            //curr->mmpBBs[cid] = newIns;
                            //포인트 추가
                            bMerge = true;
                            strLabel = prevIns->mStrLabel;
                        }
                        /*else 
                            std::cout << "prev ins test = " << prevIns->mStrLabel << " " << currIns->mStrLabel <<" "<<currIns->mnLabel << " " << pratio << " " << cratio << " " << ratio << std::endl;*/
                    }
                    if (bNotInstance && cratio > 0.8 && cIsthing && bPrevTable && !bCurrTable) {
                        cv::Mat newRow = cv::Mat::zeros(1, maxCurr + 1, CV_32F);
                        cv::vconcat(iou_matrix, newRow, iou_matrix);
                        pid = ++maxPrev;
                        val = cratio;
                        pair.first = pid;
                        bMerge = true;
                        strLabel = currIns->mStrLabel;
                    }
                    if (bMerge) {
                        std::cout << "InstanceLinker::Add New Instance = " << strLabel <<" = "<<val << std::endl;
                    }
                    if (val > iou_threshold && !bMerge && plabel != clabel) {
                        std::cout << "case test = " << prevIns->mStrLabel << " " << currIns->mStrLabel << " == " << val << std::endl;
                    }
                }

                /*if (bPrevTable && bCurrTable)
                    std::cout << "table iou test = " <<pid<<" "<<cid<<" == "<< val << std::endl;*/

                if (val < iou_threshold)
                    val = 0.0;
                iou_matrix.at<float>(pid, cid) = val;

                if (val > 0.0) {
                    assignments[pid] = cid;
                    //assignments.insert(pair);
                }
            }
        }
        //std::cout << iou_matrix << std::endl;

        prev->mnMaxID = maxPrev;
        curr->mnMaxID = maxCurr;
    }

    void InstanceLinker::computeFromOF(BoxFrame* prev, BoxFrame* curr,
            const std::vector<std::pair<int, int>>& vecPairMatchInstance, std::map<int,int>& assignments, std::map<std::pair<int, int>, std::pair<int, int>>& mapChanged, float iou_threshold)
    {
        auto prevBBs = prev->mmpBBs;
        auto currBBs = curr->mmpBBs;

        std::map<std::pair<int, int>, int> mapLinkCount;
        std::map<int, int> mapPrevCount, mapCurrCount;

        int maxPrev = prev->mnMaxID;
        int maxCurr = curr->mnMaxID;

        for (auto pair : vecPairMatchInstance) {
            int pid = pair.first;
            int cid = pair.second;

            /*if (pid > maxPrev)
                maxPrev = pid;
            if (cid > maxCurr)
                maxCurr = cid;*/

            mapPrevCount[pid]++;
            mapCurrCount[cid]++;
            mapLinkCount[std::make_pair(pid, cid)]++;
        }

        std::vector<std::pair<int, int>> res;
        //sstd::cout << "link::2" << std::endl;
        cv::Mat iou_matrix = cv::Mat::zeros(maxPrev+1, maxCurr+1, CV_32F);
        for (auto ppair : mapPrevCount) {
            int pid = ppair.first;
            auto oriPrevID = pid;
            float pcount = (float)ppair.second;

            auto prevIns = prev->mmpBBs[pid];
            int plabel = prevIns->mpConfLabel->label;
            auto pIsthing = prevIns->mbIsthing;
            bool bPrevTable = prevIns->isTable();

            for (auto cpair : mapCurrCount) {
                int cid = cpair.first;
                auto oriCurrID = cid;
                float ccount = (float)cpair.second;
                
                auto currIns = curr->mmpBBs[cid];
                int clabel = currIns->mpConfLabel->label;
                auto cIsthing = currIns->mbIsthing;
                bool bCurrTable = currIns->isTable();
                
                auto pair = std::make_pair(pid, cid);
                float count = (float)mapLinkCount[pair];

                float sum = pcount + ccount - count;
                float val = count / sum;

                if (count < 5)
                    continue;

                {
                    
                    auto pratio = count / pcount;
                    auto cratio = count / ccount;
                    auto ratio = count / sum;
                    
                    bool bMerge = false;
                    bool bNotInstance = ratio < iou_threshold;
                    
                    //curr에 추가
                    //다른 데이터가 있어서 iou로 직접 연결이 안되는 경우에 자기 자신의 인스턴스는 대부분 다른 곳에 속할 때 물체로 인식되고, 한쪽은 테이블, 한쪽은 테이블이 아니면 물체가 인식이 안되었다고 판단하기.
                    //if (bNotInstance && pratio > 0.8 && pIsthing && bCurrTable && !bPrevTable) {
                    //    //일단 다이닝 테이블인 경우 추가 60
                    //    //새로운 인스턴스를 추가해야 함.
                    //    cv::Mat newCol = cv::Mat::zeros(maxPrev + 1, 1, CV_32F);
                    //    cv::hconcat(iou_matrix, newCol, iou_matrix);

                    //    cid = ++maxCurr;
                    //    val = pratio;
                    //    pair.second = cid;
                    //    bMerge = true;
                    //    //std::cout << "InstanceLinker::Add New Instance = " << prevIns->mStrLabel<< std::endl;
                    //    //auto newIns = new SegInstance(curr, curr->fx, curr->fy, curr->cx, curr->cy, plabel, prevIns->mfConfidence, prevIns->mbIsthing, nullptr);
                    //    //curr->mmpBBs[cid] = newIns;
                    //    
                    //    //포인트 추가
                    //}

                    ////curr에 추가
                    if (bNotInstance && pratio > 0.9 && pIsthing && bCurrTable && !bPrevTable) {//prevIns->mbDetected && 
                        cv::Mat newCol = cv::Mat::zeros(maxPrev + 1, 1, CV_32F);
                        cv::hconcat(iou_matrix, newCol, iou_matrix);
                        cid = ++maxCurr;
                        val = pratio;
                        pair.second = cid;
                        bMerge = true;
                    }

                    ////prev에 추가
                    if (bNotInstance && cratio > 0.9 && cIsthing && bPrevTable && !bCurrTable) {//currIns->mbDetected && 
                        cv::Mat newRow = cv::Mat::zeros(1, maxCurr + 1, CV_32F);
                        cv::vconcat(iou_matrix, newRow, iou_matrix);
                        pid = ++maxPrev;
                        val = cratio;
                        pair.first = pid;
                        bMerge = true;
                    }
                    if (bMerge) {
                        mapChanged[std::make_pair(oriPrevID, oriCurrID)] = std::make_pair(pid, cid);
                    }
                }

                if (val < iou_threshold)
                    val = 0.0;
                iou_matrix.at<float>(pid, cid) = val;

                if (val > 0.0) {
                    assignments[pid] = cid;
                    //assignments.insert(pair);
                }     
            }
        }
        prev->mnMaxID = maxPrev;
        curr->mnMaxID = maxCurr;
    }

    std::vector<std::pair<int, int>> InstanceLinker::computeFromOF(BoxFrame* prev, BoxFrame* curr,
        const std::vector<cv::Point2f>& vecPrevCorners, const std::vector<cv::Point2f>& vecCurrCorners, const std::vector<int>& vecMatchingIDXs, float iou_threshold)
    {
        auto prevBBs = prev->mmpBBs;
        auto currBBs = curr->mmpBBs;

        int maxPrev = 0;
        int maxCurr = 0;
        std::map<std::pair<int, int>, int> mapLinkCount;
        std::map<int, int> mapPrevCount, mapCurrCount;

        std::vector<std::pair<int, int>> res;
        std::cout << "link::1" << std::endl;
        for (int i = 0; i < vecMatchingIDXs.size(); i++) {
            int idx = i;
            auto prevPt = vecPrevCorners[idx];
            auto currPt = vecCurrCorners[idx];

            //std::cout << prevPt << " " << currPt << std::endl;

            if (currPt.x < 0.0 || currPt.y < 0.0 || prevPt.x < 0.0 || prevPt.y < 0.0)
            {
                std::cout << "link::err::position error::"<<prev->mnId<<" "<<curr->mnId<<"==" << currPt << prevPt << std::endl;
            }

            int pid = prev->GetInstance(prevPt);
            int cid = curr->GetInstance(currPt);

            if (pid < 0 || pid > 200 || cid < 0 || cid > 200)
            {
                std::cout << "link::err::id::" << " " << pid << " " << cid <<" "<< std::endl;

            }

            if (pid > maxPrev)
                maxPrev = pid;
            if (cid > maxCurr)
                maxCurr = cid;

            mapPrevCount[pid]++;
            mapCurrCount[cid]++;
            mapLinkCount[std::make_pair(pid, cid)]++;
            
            //int prevInsId = prev->sinfos[pid].at<float>(0);
            //iou_matrix.at<float>(pid, cid)++;
        }
        std::cout << "link::2" << std::endl;
        //cv::Mat iou_matrix = cv::Mat::zeros(maxPrev, maxCurr, CV_32F);
        //for (auto ppair : mapPrevCount) {
        //    int pid = ppair.first;
        //    float pcount = (float)ppair.second;
        //    for (auto cpair : mapCurrCount) {
        //        int cid = cpair.first;
        //        float ccount = (float)cpair.second;

        //        auto pair = std::make_pair(pid, cid);
        //        float count = (float)mapLinkCount[pair];
        //        float sum = pcount + ccount - count;
        //        iou_matrix.at<float>(pid, cid) = count / sum;
        //    }
        //}
        //std::cout << "link::3 = " << std::endl;
        //std::vector<int> assignments = hungarianMatching(iou_matrix);
        //std::cout << "link::4" << std::endl;
        ////link
        //for (int i = 0; i < assignments.size(); ++i) {
        //    int j = assignments[i];
        //    if (j != -1 && iou_matrix.at<float>(i, j) > iou_threshold) {
        //        res.push_back(std::make_pair(i, j));
        //    }
        //}
        //std::cout << "link::5" << std::endl;
        return res;
    }
	void InstanceLinker::compute(BoxFrame* prev, BoxFrame* curr, float iou_threshold) {
		std::map<int, std::vector<std::pair<int, int>>> global_instances;
		int next_global_id = 0;
		auto prevBBs = prev->mmpBBs;
		auto currBBs = curr->mmpBBs;

		cv::Mat iou_matrix = computeIoUMatrixWithMPs(prevBBs, currBBs);
        std::vector<int> assignments;
        hungarianMatching(iou_matrix , assignments);

        for (int i = 0; i < assignments.size(); ++i) {
            int j = assignments[i];
            if (j != -1 && iou_matrix.at<float>(i, j) > iou_threshold) {
                // Find the global ID for the matched previous instance
                for (auto& [global_id, instance_list] : global_instances) {
                    /*if (!instance_list.empty() &&
                        instance_list.back().first == frame_idx - 1 &&
                        instance_list.back().second == j) {
                        instance_list.push_back({ frame_idx, i });
                        break;
                    }*/
                }
            }
            else {
                // If no match found, create a new global instance
                //global_instances[next_global_id] = { {frame_idx, i} };
                //++next_global_id;
            }
        }
	}

	cv::Mat InstanceLinker::computeIoUMatrixWithMPs(const std::map<int, SegInstance*>& instances1, const std::map<int, SegInstance*>& instances2) {
		cv::Mat iou_matrix(instances1.size(), instances2.size(), CV_32F);
		return iou_matrix;
	}

    cv::Mat InstanceLinker::computeIoUMatrixWithKPs(const std::map<int, SegInstance*>& instances1, const std::map<int, SegInstance*>& instances2)
    {
    
    }

    void InstanceLinker::hungarianMatching(const cv::Mat& cost_matrix, std::vector<int>& assignment) {
        cv::Mat cost_matrix_int;
        cost_matrix.convertTo(cost_matrix_int, CV_32S, -100000);  // Convert to cost (negative of score)
        HungarianMatcher::compute(cost_matrix_int, assignment);
    }

}   