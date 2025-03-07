#include <InstanceLinker.h>

#include <SegInstance.h>
#include <BoxFrame.h>
#include <FrameInstance.h>
#include <GlobalInstance.h>
#include <KeyFrame.h>
#include <Frame.h>
#include <MapPoint.h>
#include <SemanticLabel.h>

#include <ObjectSLAM.h>
#include <Utils_Geometry.h>

namespace ObjectSLAM {

    ObjectSLAM* InstanceLinker::ObjectSystem = nullptr;

    float InstanceSim::CalculateIOU(const cv::Rect& rect1, const cv::Rect& rect2) {
        // 교집합 영역 계산
        int x1 = std::max(rect1.x, rect2.x);
        int y1 = std::max(rect1.y, rect2.y);
        int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
        int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

        // 교집합이 없는 경우
        if (x2 < x1 || y2 < y1)
            return 0.0f;

        // 교집합 면적 계산
        float intersection_area = (x2 - x1) * (y2 - y1);

        // 각 사각형의 면적 계산
        float area1 = rect1.width * rect1.height;
        float area2 = rect2.width * rect2.height;

        // 합집합 면적 계산 (area1 + area2 - intersection_area)
        float union_area = area1 + area2 - intersection_area;

        // IOU 계산
        return intersection_area / union_area;
    }

    std::pair<bool, cv::Point2f> InstanceSim::ConvertFlowPoint(const cv::Mat& flow, const cv::Point2f& src) {

        auto rpt = src / 4;

        bool bres = false;
        auto dst = cv::Point2f(-1, -1);
        if (rpt.x < 0 || rpt.y < 0 || rpt.x >= flow.cols || rpt.y >= flow.rows)
        {
            auto res = std::make_pair(bres, dst);
            return res;
        }

        cv::Vec<schar, 2> tmp = flow.at<cv::Vec<schar, 2>>(rpt) * 4;

        if (tmp.val[0] == 0 && tmp.val[1] == 0) {
            dst.x =  -2;
            dst.y = -2;
            auto res = std::make_pair(bres, dst);
            return res;
        }
        /*
        dst.x = src.x+tmp.val[0];
        dst.y = src.y+tmp.val[1];
        if (dst.x < 0 || dst.x >= flow.cols*4 || dst.y < 0 || dst.y >= flow.rows*4)
        {
            std::cout << "flow err bound = " << dst << std::endl;
            dst.x = -1;
            dst.y = -1;
            auto res = std::make_pair(bres, dst);
            return res;
        }
        */

        bres = true;
        dst.x = tmp.val[0];
        dst.y = tmp.val[1];
        auto res = std::make_pair(bres, dst);
        return res;
    }

    bool InstanceSim::ComputeRaftInstance(const cv::Mat& flow, FrameInstance* pPrev, FrameInstance* pCurr) {
        //cetner
        //rect
        //contour
        auto rect = pPrev->rect;
        auto pt = pPrev->pt;
        auto contour = pPrev->contour;
        cv::Point2f npt;
        std::vector<cv::Point2f> ncontour;
        std::vector<std::pair<bool, cv::Point2f>> res;

        res.push_back((ConvertFlowPoint(flow, pt)));

        /*for(auto cpt : contour){
            res.push_back((ConvertFlowPoint(flow, cpt)));
        }*/

        int ntest = 0;
        /*int nfail1 = 0;
        int nfail2 = 0;
        for (auto pair : res)
        {
            auto b = pair.first;
            auto rpt = pair.second;
            if (b)
                ntest++;
            else {
                if (rpt.x == -1)
                    nfail1++;
                else if (rpt.x == -2)
                    nfail2++;
            }
        }*/

        if (!res[0].first)
        {
            //average point
            //반이상 변환이 되면 나머지를 그 중심으로 변환.
            auto tpt = res[0].second;
            std::cout << "fail raft flow = " << pt.x << std::endl << std::endl << std::endl;;

            std::vector<cv::Point2f> pts;
            pts.push_back(cv::Point2f(pt.x - 4, pt.y));
            pts.push_back(cv::Point2f(pt.x + 4, pt.y));
            pts.push_back(cv::Point2f(pt.x, pt.y - 4));
            pts.push_back(cv::Point2f(pt.x, pt.y + 4));

            int nfail = 0;
            for (auto tpt : pts)
            {
                auto tres = ConvertFlowPoint(flow, tpt);
                if (tres.first)
                {
                    tpt.x += tres.second.x;
                    tpt.y += tres.second.y;
                }
                else {
                    nfail++;
                }
            }

            //for (auto pt : contour) {
            //    auto npt = pt;
            //    //npt.x += rpt.x;
            //    //npt.y += rpt.y;

            //    auto tres = ConvertFlowPoint(flow, pt);
            //    if (tres.first)
            //    {
            //        npt.x += tres.second.x;
            //        npt.y += tres.second.y;
            //    }
            //    else {
            //        nfail++;
            //    }
            //    pCurr->contour.push_back(npt);
            //}
            std::cout << "raft fail test = " << nfail << " " << pts.size() << std::endl << std::endl;
            return false;
        }
        else
        {
            auto rpt = res[0].second;
            pCurr->pt = pt + rpt;
            rect.x += rpt.x;
            rect.y += rpt.y;
            pCurr->rect = rect;
            for (auto pt : contour) {
                auto npt = pt;
                npt.x += rpt.x;
                npt.y += rpt.y;
                pCurr->contour.push_back(npt);
            }

            pCurr->area = pPrev->area;

            //mask
            std::vector<std::vector<cv::Point>> contours;
            contours.push_back(pCurr->contour);
            cv::Mat newmask = cv::Mat::zeros(flow.rows*4, flow.cols*4, CV_8UC1);;
            cv::drawContours(newmask, contours, 0, cv::Scalar(255, 255, 255), -1);
            pCurr->mask = newmask.clone();
            return true;
        }
        //if(ntest != res.size())
            //std::cout << "flow test = " <<res[0].first<<" == " << ntest << " " << res.size() << "==" << nfail1<<", "<<nfail2<< std::endl;

        return false;
    }

    bool InstanceSim::CheckStaticObject(const std::vector<cv::Point>& contour, std::map<int, FrameInstance*>& mapInstances, int th) {

        float n = mapInstances.size();
        if (n == 0)
            return false;
        float c = 0;
        for (auto pair : mapInstances)
        {
            if (pair.first == 0)
                continue;
            auto pt = pair.second->pt;
            if (cv::pointPolygonTest(contour, pt, false) < 0.0)
                continue;
            c++;
        }
        return c >= th;
    }

    bool InstanceSim::ComputSim(const std::vector<cv::Point>& contour, const std::vector<cv::Point2f>& pts, float& val, float th) {
        float n = pts.size();
        if (n == 0){
            val = 0.0;
            return false;
        }
        float c = 0;
        for (auto pt : pts)
        {
            if (cv::pointPolygonTest(contour, pt, false) < 0.0)
                continue;
            c++;
        }
        val = c / n;
        return val >= th;
    }

    void InstanceSim::FindOverlapMP(FrameInstance* a, FrameInstance* b, std::set<EdgeSLAM::MapPoint*>& setMPs){
        for (auto mp1 : a->setMPs)
        {
            if (!mp1 || mp1->isBad())
                continue;
            for (auto mp2 : b->setMPs)
            {
                if (!mp2 || mp2->isBad())
                    continue;
                if (mp1 == mp2) {
                    setMPs.insert(mp1);
                }
            }
        }
    }
    void InstanceSim::FindOverlapMP(FrameInstance* a, EdgeSLAM::Frame* pF, std::set<EdgeSLAM::MapPoint*>& setMPs) {

    }
    void InstanceSim::FindOverlapMP(FrameInstance* a, EdgeSLAM::KeyFrame* pKF, std::set<EdgeSLAM::MapPoint*>& setMPs) {
        for (auto mp1 : a->setMPs)
        {
            if (!mp1 || mp1->isBad())
                continue;
            if (mp1->IsInKeyFrame(pKF))
            {
                setMPs.insert(mp1);
            }
        }
    }
    float InstanceSim::ComputeSimFromPartialMP(FrameInstance* a, EdgeSLAM::Frame* b){
        
        float nA = a->setMPs.size();
        float nC = 0;
        for (auto mp1 : a->setMPs)
        {
            if (!mp1 || mp1->isBad())
                continue;
        }
        return 0.0;
    }
    float InstanceSim::ComputeSimFromPartialMP(FrameInstance* a, EdgeSLAM::KeyFrame* pKF){
        float nA = a->setMPs.size();
        float nC = 0;
        for (auto mp1 : a->setMPs)
        {
            if (!mp1 || mp1->isBad())
                continue;
            if (mp1->IsInKeyFrame(pKF))
            {
                nC++;
            }
        }
        float res = nC / nA;
        if (nA == 0)
            res = 0.0;
        return res;
    }
    float InstanceSim::ComputeSimFromPartialMP(FrameInstance* a, FrameInstance* b) {
        float nA = a->setMPs.size();
        float nC = 0;
        for (auto mp1 : a->setMPs)
        {
            if (!mp1 || mp1->isBad())
                continue;
            for (auto mp2 : b->setMPs)
            {
                if (!mp2 || mp2->isBad())
                    continue;
                if (mp1 == mp2) {
                    nC++;
                }
            }
        }
        float res = nC / nA;
        if (nA == 0)
            res = 0.0;
        //std::cout << "sim test = " << res << " " << nC << std::endl;
        return res;
    }
    float InstanceSim::ComputeSimFromMP(FrameInstance* a, FrameInstance* b) {
        float nA = a->setMPs.size();
        float nB = b->setMPs.size();
        float nC = 0;
        for (auto mp1 : a->setMPs)
        {
            if (!mp1 || mp1->isBad())
                continue;
            for (auto mp2 : b->setMPs)
            {
                if (!mp2 || mp2->isBad())
                    continue;
                if (mp1 == mp2) {
                    nC++;
                }
            }
        }
        float nSum = nA + nB - nC;
        float res = nC / nSum;
        if (nSum == 0)
            res = 0.0;
        //std::cout << "sim test = " << res <<" "<<nC << std::endl;
        return res;
    }

    float InstanceSim::ComputeSimFromIOU(const cv::Mat& mask1, const cv::Mat& mask2) {
        cv::Mat overlap = mask1 & mask2;
        cv::Mat total = mask1 | mask2;
        float nOverlap = (float)cv::countNonZero(overlap);
        float nUnion = (float)cv::countNonZero(total);
        if (nUnion == 0)
            nUnion++;
        return nOverlap / nUnion;
    }

    void InstanceLinker::SetSystem(ObjectSLAM* p) {
        ObjectSystem = p;
    }

    void InstanceLinker::FindInstances(BoxFrame* pNewBF, BoxFrame* pPrevBF, std::set<int>& setMatchIDs, std::vector<cv::Point2f>& vecPoints, bool bShow) {
        std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();
        auto pCurrSeg = pNewBF->mapMasks.Get("yoloseg");
        auto pKF = pNewBF->mpRefKF;
        auto pCurrSegInstances = pCurrSeg->FrameInstances.Get();
        auto vpNeighBFs = ObjectSystem->GetConnectedBoxFrames(pKF, 10);
        auto vpNeighKFs = pKF->GetBestCovisibilityKeyFrames(10);

        cv::Mat K = pKF->K.clone();
        cv::Mat T = pKF->GetPose();
        cv::Mat R = pKF->GetRotation();
        cv::Mat t = pKF->GetTranslation();

        int ntest = 0;
        std::set<BoxFrame*> spNeighBFs;
        std::set<GlobalInstance*> spNeighGlobalIns;
        for (auto pBF : vpNeighBFs) {

            if (pBF == pPrevBF)
                continue;

            if (pBF->mapMasks.Count("yoloseg"))
            {
                spNeighBFs.insert(pBF);

                auto mapGlobals = pBF->mapMasks.Get("yoloseg")->MapInstances.Get();
                for (auto pair : mapGlobals)
                {
                    auto pG = pair.second;
                    if (!pG)
                        continue;
                    if (!spNeighGlobalIns.count(pG) && !setMatchIDs.count(pG->mnId))
                        spNeighGlobalIns.insert(pG);
                }
            }
        }
        if(bShow)
            std::cout << "missing global " << spNeighGlobalIns.size() << std::endl;
        int nTotal = 0;
        for (auto pG : spNeighGlobalIns) {
            auto spMPs = pG->AllMapPoints.Get();
            int nCount = 0;
            for (auto pMP : spMPs)
            {
                if (!pMP || pMP->isBad())
                    continue;
                if (pMP->IsInKeyFrame(pKF))
                {
                    nCount++;
                }
            }
            if (nCount > 5)
            {
                auto pt = pG->ProjectPoint(T, K);
                if (pt.x > 0 && pt.x < 640 && pt.y > 0 && pt.y < 480)
                {
                    vecPoints.push_back(pt);
                    nTotal++;
                }
            }
            if(bShow && nCount > 5)
                std::cout << "test global = " << pG->mnId << " " << nCount <<" == "<< pG->mapConnected.Size()<<","<<spMPs.size() << std::endl;
            if (nTotal > 5)
                break;
        }

        if(false)
        for (auto apPrevBF : vpNeighBFs) {
            if (!apPrevBF->mapMasks.Count("yoloseg"))
                continue;
            auto pPrevSeg = apPrevBF->mapMasks.Get("yoloseg");
            auto pPrevInstances = pPrevSeg->FrameInstances.Get();

            for (auto pair : pPrevInstances)
            {
                if (pair.first == 0)
                    continue;
                auto pPrevIns = pair.second;
                std::set<EdgeSLAM::MapPoint*> setMPs;
                float nMP = pPrevIns->setMPs.size();
                InstanceSim::FindOverlapMP(pPrevIns, pKF, setMPs);

                if (setMPs.size() > 5)
                {
                    cv::Point2f avgPt(0, 0);
                    int n = 0;
                    for (auto pMP : setMPs)
                    {
                        if (!pMP || pMP->isBad())
                            continue;
                        cv::Point2f pt;
                        float depth;
                        CommonUtils::Geometry::ProjectPoint(pt, depth, pMP->GetWorldPos(), K, R, t);
                        if (depth > 0.0)
                        {
                            n++;
                            avgPt += pt;
                        }
                    }
                    if (n > 0)
                    {
                        avgPt /= n;
                        vecPoints.push_back(avgPt);
                    }
                    
                    auto pG = pPrevSeg->MapInstances.Get(pair.first);
                    //auto pG = pPrevIns->mpGlobal;
                    if (pG && pG->mapConnected.Count(pNewBF)) {
                        int idx = pG->mapConnected.Get(pNewBF);
                        if (!setMatchIDs.count(idx))
                        {
                            
                        }
                    }
                    ntest++;
                    //std::cout << "test = " << setMPs.size() << " " << nMP << std::endl;
                }
            }
        }
        if (bShow)
        {
            std::chrono::high_resolution_clock::time_point t_end = std::chrono::high_resolution_clock::now();
            auto du_seg = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

            std::cout << "test ~~~~~~~~~~"<< pNewBF->mnId<<" == " << du_seg << ", " << vecPoints.size() << " == " << pCurrSegInstances.size() - setMatchIDs.size() << std::endl;
        }
    }

    void InstanceLinker::LinkNeighBFs(BoxFrame* pNewBF, std::set<int>& setMatchIDs, bool bShow) {

        std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();
        auto pCurrSeg = pNewBF->mapMasks.Get("yoloseg");
        auto pKF = pNewBF->mpRefKF;
        auto pCurrSegInstances = pCurrSeg->FrameInstances.Get();
        
        auto vpNeighBFs = ObjectSystem->GetConnectedBoxFrames(pKF, 10);

        //global
        int ntest = 0;
        for (auto pair : pCurrSegInstances) {
            int sid = pair.first;
            auto pCurrIns = pair.second;

            bool bMatch = false;

            if (setMatchIDs.count(sid))
                continue;
            for (auto pPrevBF : vpNeighBFs) {
                if (!pPrevBF->mapMasks.Count("yoloseg"))
                    continue;
                auto pPrevSeg = pPrevBF->mapMasks.Get("yoloseg");
                auto pPrevInstances = pPrevSeg->FrameInstances.Get();

                for (auto pair : pPrevInstances)
                {
                    if (pair.first == 0)
                        continue;
                    auto pPrevIns = pair.second;
                    float val = InstanceSim::ComputeSimFromMP(pCurrIns, pPrevIns);
                    //std::cout << "tes = " << val << std::endl;
                    if (val > 0.5)
                    {
                        bMatch = true;
                        if (bMatch)
                            break;
                    }
                }
                if (bMatch)
                    break;
            }
            if (bMatch){
                ntest++;
                setMatchIDs.insert(sid);
                break;
            }
        }

        if (bShow)
        {
            std::chrono::high_resolution_clock::time_point t_end = std::chrono::high_resolution_clock::now();
            auto du_seg = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

            std::cout << "test ~~~~~~~~~~~~~~~ ==  " << du_seg <<", "<<ntest << "  ==  " << pCurrSegInstances.size() - setMatchIDs.size() << std::endl;
        }
        
    }

    void InstanceLinker::computeSim(BoxFrame* prev, BoxFrame* curr, const std::vector<std::pair<int, int>>& vecPairMatches, float iou_threshold) {


        std::map<std::pair<int, int>, float> mapLinkCount;
        std::map<int, float> mapPrevCount, mapCurrCount;
        std::map<int, std::vector<int>> mapPrevInsPoints, mapCurrInsPoints;

        for (auto pair : vecPairMatches) {
            int idx1 = pair.first;
            int idx2 = pair.second;

            auto sid1 = prev->mvnInsIDs[idx1];
            auto sid2 = curr->mvnInsIDs[idx2];
            
            if (sid1 > 0) {
                mapPrevCount[sid1]++;
                mapPrevInsPoints[sid1].push_back(idx1);
            }
            if (sid2 > 0) {
                mapCurrCount[sid2]++;
                mapCurrInsPoints[sid2].push_back(idx2);
            }
            if (sid1 > 0 && sid2 > 0) {
                auto pairIns = std::make_pair(sid1, sid2);
                mapLinkCount[pairIns]++;
            }
        }

        for (auto pair : mapLinkCount) {
            int idx1 = pair.first.first;
            int idx2 = pair.first.second;

            auto label1 = prev->mmpBBs[idx1]->mStrLabel;
            auto label2 = curr->mmpBBs[idx2]->mStrLabel;

            float count = pair.second;
            float c1 = mapPrevCount[idx1];
            float c2 = mapCurrCount[idx2];

            float sum = c1 + c2 - count;
            float r = count / sum;
            float r1 = count / c1;
            float r2 = count / c2;
            std::cout << "link test = " << label1 << " " << label2 << " = " << r << " == " << r1 << " " << r2 << " " << std::endl;
        }

    }
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