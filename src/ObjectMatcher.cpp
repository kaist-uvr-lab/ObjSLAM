#include <ObjectMatcher.h>
#include <Utils.h>
#include <Utils_Geometry.h>
#include <Frame.h>
#include <KeyFrame.h>
#include <FrameInstance.h>
#include <ObjectMatchingInfo.h>

namespace ObjectSLAM {

	const  int ObjectMatcher::HISTO_LENGTH = 30;

	int ObjectMatcher::SearchInstance(const cv::Mat& desc1, const cv::Mat& desc2
		, std::vector<std::pair<int, int>>& vecMatches, const int thdist, const float thratio)
	{
		int nmatches = 0;
		int N1 = desc1.rows;
		int N2 = desc2.rows;

		std::vector<int> vMatchedDistance(N2, INT_MAX);
		std::vector<int> vMatched1(N1, -1);
		std::vector<int> vMatched2(N2, -1);

		for (int i1 = 0; i1 < N1; i1++)
		{
			cv::Mat d1 = desc1.row(i1);
			int bestDist = INT_MAX;
			int bestDist2 = INT_MAX;
			int bestIdx = -1;

			for (int i2 = 0; i2 < N2; i2++)
			{
				cv::Mat d2 = desc2.row(i2);

				int dist = (int)Utils::CalcBinaryDescriptor(d1, d2);

				if (vMatchedDistance[i2] <= dist)
					continue;

				if (dist < bestDist)
				{
					bestDist2 = bestDist;
					bestDist = dist;
					bestIdx = i2;
				}
				else if (dist < bestDist2)
				{
					bestDist2 = dist;
				}
			}
			if (bestDist <= thdist)
			{
				if (bestDist < (float)bestDist2 * thratio)
				{
					if (vMatched2[bestIdx] >= 0) {
						nmatches--;
						vMatched1[vMatched2[bestIdx]] = -1;
					}
					vMatched1[i1] = bestIdx;
					vMatched2[bestIdx] = i1;
					nmatches++;
				}
			}

		}
		for (size_t i1 = 0, iend1 = vMatched1.size(); i1 < iend1; i1++) {
			if (vMatched1[i1] >= 0) {
				int i2 = vMatched1[i1];
				vecMatches.push_back(std::make_pair(i1, i2));
			}
		}
		return vecMatches.size();
	}


	bool ObjectMatcher::removeOutliersWithMovementStatistics(
		const std::vector<cv::KeyPoint>& keypointsA,
		const std::vector<cv::KeyPoint>& keypointsB,
		const std::vector<std::pair<int, int>>& matches,
		std::vector<std::pair<int, int>>& filteredMatches,
		float stdDevThreshold) {

		if (matches.size() < 3) {
			return false; // 통계적 의미를 갖기에 데이터가 너무 적음
		}

		// 이동 벡터 계산
		std::vector<cv::Point2f> movements;
		for (const auto& match : matches) {
			cv::Point2f ptA = keypointsA[match.first].pt;
			cv::Point2f ptB = keypointsB[match.second].pt;
			movements.push_back(ptB - ptA);
		}

		// 이동 벡터의 x, y 평균 및 표준편차 계산
		cv::Point2f meanMovement(0, 0);
		for (const auto& m : movements) {
			meanMovement += m;
		}
		meanMovement *= 1.0f / movements.size();

		float sumSquaredX = 0, sumSquaredY = 0;
		for (const auto& m : movements) {
			float dx = m.x - meanMovement.x;
			float dy = m.y - meanMovement.y;
			sumSquaredX += dx * dx;
			sumSquaredY += dy * dy;
		}

		float stdDevX = std::sqrt(sumSquaredX / movements.size());
		float stdDevY = std::sqrt(sumSquaredY / movements.size());

		// 표준편차의 배수 내에 있는 매칭만 유지
		for (size_t i = 0; i < movements.size(); i++) {
			float dx = std::abs(movements[i].x - meanMovement.x);
			float dy = std::abs(movements[i].y - meanMovement.y);

			if (dx <= stdDevThreshold * stdDevX && dy <= stdDevThreshold * stdDevY) {
				filteredMatches.push_back(matches[i]);
			}
		}

		return true;
	}

	bool ObjectMatcher::removeOutliersWithMahalanobis(
		const std::vector<cv::KeyPoint>& keypointsA,
		const std::vector<cv::KeyPoint>& keypointsB,
		const std::vector<std::pair<int, int>>& matches,
		std::vector<std::pair<int, int>>& filteredMatches,
		float threshold) { // 2-DOF 카이제곱 분포에서 p=0.05에 대한 임계값

		if (matches.size() < 3) {
			return false;
		}

		// 이동 벡터 계산
		std::vector<cv::Point2f> movements;
		for (const auto& match : matches) {
			cv::Point2f ptA = keypointsA[match.first].pt;
			cv::Point2f ptB = keypointsB[match.second].pt;
			movements.push_back(ptB - ptA);
		}

		// 평균 이동 벡터
		cv::Point2f mean(0, 0);
		for (const auto& m : movements) {
			mean += m;
		}
		mean *= 1.0f / movements.size();

		// 공분산 행렬 계산
		float covXX = 0, covXY = 0, covYY = 0;
		for (const auto& m : movements) {
			float dx = m.x - mean.x;
			float dy = m.y - mean.y;
			covXX += dx * dx;
			covXY += dx * dy;
			covYY += dy * dy;
		}
		covXX /= movements.size();
		covXY /= movements.size();
		covYY /= movements.size();

		// 공분산 행렬의 역행렬 계산
		float det = covXX * covYY - covXY * covXY;
		if (std::abs(det) < 1e-10) {
			// 행렬이 특이하면 기본 방법으로 대체
			return removeOutliersWithMovementStatistics(keypointsA, keypointsB, matches, filteredMatches);
		}

		float invCovXX = covYY / det;
		float invCovXY = -covXY / det;
		float invCovYY = covXX / det;

		// 마할라노비스 거리 계산 및 필터링
		for (size_t i = 0; i < movements.size(); i++) {
			float dx = movements[i].x - mean.x;
			float dy = movements[i].y - mean.y;

			// 마할라노비스 거리의 제곱
			float mahaDist = dx * dx * invCovXX + 2 * dx * dy * invCovXY + dy * dy * invCovYY;

			if (mahaDist <= threshold) {
				filteredMatches.push_back(matches[i]);
			}
		}

		return true;
	}


	/// <summary>
	/// ////////////////////////////////////////////////////////////
	/// </summary>
	/// <param name="pTarget"></param>
	/// <param name="pRef"></param>
	/// <param name="gray1"></param>
	/// <param name="gray2"></param>
	/// <param name="pMatches"></param>
	/// <returns></returns>

	int ObjectMatcher::SearchInstance(EdgeSLAM::Frame* pTarget, BoxFrame* pRef, const cv::Mat& gray1, const cv::Mat& gray2, ObjectMatchingInfo* pMatches) {

		auto pSeg = pRef->mapMasks.Get("yoloseg");
		
		std::vector<cv::Point2f> corners;
		//cv::goodFeaturesToTrack(gray2, pMatches->vecReferenceCorners, 1000, 0.01, 10, pSeg->mask);
		
		auto pSegIns = pSeg->FrameInstances.Get();
		for (auto pair : pSegIns)
		{
			pMatches->vecReferenceCorners.push_back(pair.second->pt);
			/*std::vector<cv::Point2f> corners;
			cv::goodFeaturesToTrack(gray2, corners, 5, 0.01, 10, pair.second->mask);
			for (auto pt : corners)
			{
				pMatches->vecReferenceCorners.push_back(pt);
			}*/
			
		}

		/*for (auto pt : corners)
		{
			int label = pSeg->mask.at<uchar>(pt);
			if (label < 1)
			{
				continue;
			}
			pMatches->vecReferenceCorners.push_back(pt);
		}*/
		
		std::vector<int> vecInsIDs;
		std::vector<int> idxs;
		/*for (auto pair : pSeg->instance)
		{
			auto pIns = pair.second;
			auto label = pair.first;
			if (label > 0) {
				pMatches->vecReferenceCorners.push_back(pIns->pt);
				vecInsIDs.push_back(label);
			}
			
		}*/
		
		/*for (auto kp : pRef->mvKeyDatas)
		{
			int label = pSeg->mask.at<uchar>(kp.pt);
			if (label < 1)
			{
				continue;
			}
			pMatches->vecReferenceCorners.push_back(kp.pt);
			vecInsIDs.push_back(label);
		}*/
		
		if (pMatches->vecReferenceCorners.size() == 0)
			return 0;
		
		int win_size = 10;
		cv::calcOpticalFlowPyrLK(
			gray2,                         // Previous image
			gray1,                         // Next image
			pMatches->vecReferenceCorners,                     // Previous set of corners (from imgA)
			pMatches->vecTargetCorners,                     // Next set of corners (from imgB)
			pMatches->vecFounds,               // Output vector, each is 1 for tracked
			cv::noArray(),                // Output vector, lists errors (optional)
			cv::Size(win_size * 2 + 1, win_size * 2 + 1),  // Search window size
			5,                            // Maximum pyramid level to construct
			cv::TermCriteria(
				cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS,
				20,                         // Maximum number of iterations
				0.3                         // Minimum change per iteration
			)
		);

		auto pKF = pRef->mpRefKF;

		//에피폴라 제약을 이용한 매칭 에러 체크
		cv::Mat T1 = pTarget->GetPose();
		cv::Mat R1 = T1.rowRange(0, 3).colRange(0, 3);
		cv::Mat t1 = T1.rowRange(0, 3).col(3);
		const cv::Mat T2 = pKF->GetPose();
		cv::Mat R2 = T2.rowRange(0, 3).colRange(0, 3);
		cv::Mat t2 = T2.rowRange(0, 3).col(3);
		const cv::Mat K1 = pTarget->K.clone();
		const cv::Mat K2 = pKF->K.clone();
		cv::Mat F12 = CommonUtils::Geometry::ComputeF12(R1, t1, R2, t2, K1, K2);

		int res = 0;
		int inc = 0;
		for (int i = 0; i < pMatches->vecFounds.size(); ++i) {
			if (!pMatches->vecFounds[i]) {
				continue;
			}
			auto targetPt = pMatches->vecTargetCorners[i];
			auto referencePt = pMatches->vecReferenceCorners[i];

			//디스크립터 계산용. 
			if (referencePt.x <= inc || referencePt.x >= gray2.cols - inc || referencePt.y <= inc || referencePt.y >= gray2.rows - inc) {
				pMatches->vecFounds[i] = false;
				continue;
			}
			
			//epipolar 제약
			if (!CommonUtils::Geometry::CheckDistEpipolarLine(targetPt, referencePt, F12, 1.0)) {
				pMatches->vecFounds[i] = false;
				continue;
			}

			res++;
		}
		return res;
	}

	int ObjectMatcher::SearchByOpticalFlow(EdgeSLAM::Frame* pTarget, EdgeSLAM::KeyFrame* pRef, const cv::Mat& gray1, const cv::Mat& gray2, ObjectMatchingInfo* pMatches) {
		
		std::vector<int> idxs;
		for (int i = 0; i < pTarget->N; i++) {
			pMatches->vecTargetCorners.push_back(pTarget->mvKeys[i].pt);
		}

		if (pMatches->vecTargetCorners.size() < 10)
			return 0;

		cv::Mat mUsed = cv::Mat::zeros(gray2.size(), CV_32SC1);
		for (int i = 0; i < pRef->N; i++) {
			auto kp = pRef->mvKeys[i];
			//해당 매트릭스에 kp의 인덱스를 넣음.
			//0부터 시작이라 +1을 함. 이 매트릭스 이용시 -1 해야 함.
			mUsed.at<int>(cv::Point(kp.pt)) = i + 1;
		}


		int win_size = 10;
		cv::calcOpticalFlowPyrLK(
			gray1,                         // Previous image
			gray2,                         // Next image
			pMatches->vecTargetCorners,                     // Previous set of corners (from imgA)
			pMatches->vecReferenceCorners,                     // Next set of corners (from imgB)
			pMatches->vecFounds,               // Output vector, each is 1 for tracked
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
		cv::Mat T1 = pTarget->GetPose();
		cv::Mat R1 = T1.rowRange(0, 3).colRange(0, 3);
		cv::Mat t1 = T1.rowRange(0, 3).col(3);
		const cv::Mat T2 = pRef->GetPose();
		cv::Mat R2 = T2.rowRange(0, 3).colRange(0, 3);
		cv::Mat t2 = T2.rowRange(0, 3).col(3);
		const cv::Mat K1 = pTarget->K.clone();
		const cv::Mat K2 = pRef->K.clone();
		cv::Mat F12 = CommonUtils::Geometry::ComputeF12(R1, t1, R2, t2, K1, K2);

		int res = 0;
		int inc = 0;
		for (int i = 0; i < pMatches->vecFounds.size(); ++i) {
			if (!pMatches->vecFounds[i]) {
				continue;
			}
			auto targetPt = pMatches->vecTargetCorners[i];
			auto referencePt = pMatches->vecReferenceCorners[i];

			auto kp = pTarget->mvKeys[i];

			//디스크립터 계산용. 
			if (referencePt.x <= inc || referencePt.x >= gray2.cols - inc || referencePt.y <= inc || referencePt.y >= gray2.rows - inc){
				pMatches->vecFounds[i] = false;
				continue;
			}
			//epipolar 제약
			if (!CommonUtils::Geometry::CheckDistEpipolarLine(targetPt, referencePt, F12, pTarget->mvLevelSigma2[kp.octave])) {
				pMatches->vecFounds[i] = false;
				continue;
			}

			int idx2 = mUsed.at<int>(cv::Point(referencePt));
			if (idx2 > 0) {
				pMatches->vecTargetIdxs.push_back(i);
				pMatches->vecRefIdxs.push_back(--idx2);
			}

			res++;
		}
		return res;
	}
	int ObjectMatcher::SearchByOpticalFlow(EdgeSLAM::KeyFrame* pTarget, EdgeSLAM::KeyFrame* pRef, const cv::Mat& gray1, const cv::Mat& gray2, ObjectMatchingInfo* pMatches) {

		std::vector<int> idxs;
		for (int i = 0; i < pTarget->N; i++) {
			pMatches->vecTargetCorners.push_back(pTarget->mvKeys[i].pt);
		}

		if (pMatches->vecTargetCorners.size() < 10)
			return 0;

		cv::Mat mUsed = cv::Mat::zeros(gray2.size(), CV_32SC1);
		for (int i = 0; i < pRef->N; i++) {
			auto kp = pRef->mvKeys[i];
			//해당 매트릭스에 kp의 인덱스를 넣음.
			//0부터 시작이라 +1을 함. 이 매트릭스 이용시 -1 해야 함.
			mUsed.at<int>(cv::Point(kp.pt)) = i + 1;
		}

		int win_size = 10;
		cv::calcOpticalFlowPyrLK(
			gray1,                         // Previous image
			gray2,                         // Next image
			pMatches->vecTargetCorners,                     // Previous set of corners (from imgA)
			pMatches->vecReferenceCorners,                     // Next set of corners (from imgB)
			pMatches->vecFounds,               // Output vector, each is 1 for tracked
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
		cv::Mat T1 = pTarget->GetPose();
		cv::Mat R1 = T1.rowRange(0, 3).colRange(0, 3);
		cv::Mat t1 = T1.rowRange(0, 3).col(3);
		const cv::Mat T2 = pRef->GetPose();
		cv::Mat R2 = T2.rowRange(0, 3).colRange(0, 3);
		cv::Mat t2 = T2.rowRange(0, 3).col(3);
		const cv::Mat K1 = pTarget->K.clone();
		const cv::Mat K2 = pRef->K.clone();
		cv::Mat F12 = CommonUtils::Geometry::ComputeF12(R1, t1, R2, t2, K1, K2);

		int res = 0;
		int inc = 0;
		for (int i = 0; i < pMatches->vecFounds.size(); ++i) {
			if (!pMatches->vecFounds[i]) {
				continue;
			}
			auto targetPt = pMatches->vecTargetCorners[i];
			auto referencePt = pMatches->vecReferenceCorners[i];

			auto kp = pTarget->mvKeys[i];

			//디스크립터 계산용. 
			if (referencePt.x <= inc || referencePt.x >= gray2.cols - inc || referencePt.y <= inc || referencePt.y >= gray2.rows - inc) {
				pMatches->vecFounds[i] = false;
				continue;
			}
			//epipolar 제약
			if (!CommonUtils::Geometry::CheckDistEpipolarLine(targetPt, referencePt, F12, pTarget->mvLevelSigma2[kp.octave])) {
				pMatches->vecFounds[i] = false;
				continue;
			}
			
			int idx2 = mUsed.at<int>(cv::Point(referencePt));
			if (idx2 > 0) {
				pMatches->vecTargetIdxs.push_back(i);
				pMatches->vecRefIdxs.push_back(--idx2);
			}

			res++;
		}
		return res;
	}

	int ObjectMatcher::SearchInsAndIns(FrameInstance* pF1, FrameInstance* pF2, std::vector<std::pair<int, int>>& vecMatches, const int thdist, const float thratio, bool bCheckOri) {
		int nmatches = 0;
		int N1 = pF1->mvKeys.size();
		int N2 = pF1->mvKeys.size();

		std::vector<int> vMatchedDistance(N2, INT_MAX);
		std::vector<int> vMatched1(N1, -1);
		std::vector<int> vMatched2(N2, -1);

		for (int i1 = 0; i1 < N1; i1++)
		{
			auto kp1 = pF1->mvKeys[i1];
			int level1 = kp1.octave;

			cv::Mat d1 = pF1->mDescriptor.row(i1);
			int bestDist = INT_MAX;
			int bestDist2 = INT_MAX;
			int bestIdx = -1;

			for (int i2 = 0; i2 < N2; i2++)
			{
				cv::Mat d2 = pF2->mDescriptor.row(i2);

				int dist = (int)Utils::CalcBinaryDescriptor(d1, d2);

				if (vMatchedDistance[i2] <= dist)
					continue;

				if (dist < bestDist)
				{
					bestDist2 = bestDist;
					bestDist = dist;
					bestIdx = i2;
				}
				else if (dist < bestDist2)
				{
					bestDist2 = dist;
				}
			}
			if (bestDist <= thdist)
			{
				if (bestDist < (float)bestDist2 * thratio)
				{
					if (vMatched2[bestIdx] >= 0) {
						nmatches--;
						vMatched1[vMatched2[bestIdx]] = -1;
					}
					vMatched1[i1] = bestIdx;
					vMatched2[bestIdx] = i1;
					nmatches++;
				}
			}

		}
		for (size_t i1 = 0, iend1 = vMatched1.size(); i1 < iend1; i1++) {
			if (vMatched1[i1] >= 0) {
				int i2 = vMatched1[i1];
				vecMatches.push_back(std::make_pair(i1, i2));
			}
		}
		return vecMatches.size();
	}

	int ObjectMatcher::SearchFrameAndFrame(BoxFrame* pF1, BoxFrame* pF2, std::vector<std::pair<int, int>>& vecMatches, const float thRadius, const int thdist, const float thratio, bool bCheckOri) {
		int nmatches = 0;
		int N1 = pF1->mvKeyDatas.size();
		int N2 = pF1->mvKeyDatas.size();

		std::vector<int> vMatchedDistance(N2, INT_MAX);
		std::vector<int> vMatched1(N1, -1);
		std::vector<int> vMatched2(N2, -1);

		for (int i1 = 0; i1 < N1; i1++)
		{
			auto kp1 = pF1->mvKeyDatas[i1];
			int level1 = kp1.octave;

			cv::Mat d1 = pF1->mDescriptors.row(i1);
			int bestDist = INT_MAX;
			int bestDist2 = INT_MAX;
			int bestIdx2 = -1;

			std::vector<size_t> vIndices2 = pF2->GetFeaturesInArea(kp1.pt.x, kp1.pt.y, thRadius, level1, level1);
			if (vIndices2.empty())
				continue;

			for (std::vector<size_t>::iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
			{
				size_t i2 = *vit;
				cv::Mat d2 = pF2->mDescriptors.row(i2);

				int dist = (int)Utils::CalcBinaryDescriptor(d1, d2);

				if (vMatchedDistance[i2] <= dist)
					continue;

				if (dist < bestDist)
				{
					bestDist2 = bestDist;
					bestDist = dist;
					bestIdx2 = i2;
				}
				else if (dist < bestDist2)
				{
					bestDist2 = dist;
				}
			}
			if (bestDist <= thdist)
			{
				if (bestDist < (float)bestDist2 * thratio)
				{
					if (vMatched2[bestIdx2] >= 0) {
						nmatches--;
						vMatched1[vMatched2[bestIdx2]] = -1;
					}
					vMatched1[i1] = bestIdx2;
					vMatched2[bestIdx2] = i1;
					nmatches++;
				}
			}
		}
		for (size_t i1 = 0, iend1 = vMatched1.size(); i1 < iend1; i1++) {
			if (vMatched1[i1] >= 0) {
				int i2 = vMatched1[i1];
				vecMatches.push_back(std::make_pair(i1, i2));
			}
		}
		return vecMatches.size();
	}

	int ObjectMatcher::SearchBoxAndBox(BoundingBox* pB1, BoundingBox* pB2, std::vector<std::pair<int, int>>& vecMatches, const float thRadius, const int thdist, const float thratio, bool bCheckOri){
		int nmatches = 0;
		int N1 = pB1->N;
		int N2 = pB2->N;

		std::vector<int> vMatchedDistance(N2, INT_MAX);
		std::vector<int> vMatched1(N1, -1);
		std::vector<int> vMatched2(N2, -1);

		for (int i1 = 0; i1 < N1; i1++)
		{
			auto kp1 = pB1->mvKeyDatas[i1];
			int level1 = kp1.octave;
			cv::Mat d1 = pB1->mDescriptors.row(i1);
			int bestDist = INT_MAX;
			int bestDist2 = INT_MAX;
			int bestIdx2 = -1;

			std::vector<size_t> vIndices2 = pB2->GetFeaturesInArea(kp1.pt.x, kp1.pt.y, thRadius, level1, level1);
			if (vIndices2.empty())
				continue;

			for (std::vector<size_t>::iterator vit = vIndices2.begin(); vit != vIndices2.end(); vit++)
			{
				size_t i2 = *vit;
				cv::Mat d2 = pB2->mDescriptors.row(i2);

				int dist = (int)Utils::CalcBinaryDescriptor(d1, d2);

				if (vMatchedDistance[i2] <= dist)
					continue;

				if (dist < bestDist)
				{
					bestDist2 = bestDist;
					bestDist = dist;
					bestIdx2 = i2;
				}
				else if (dist < bestDist2)
				{
					bestDist2 = dist;
				}
			}
			if (bestDist <= thdist)
			{
				if (bestDist < (float)bestDist2 * thratio)
				{
					if (vMatched2[bestIdx2] >= 0) {
						nmatches--;
						vMatched1[vMatched2[bestIdx2]] = -1;
					}
					vMatched1[i1] = bestIdx2;
					vMatched2[bestIdx2] = i1;
					nmatches++;
				}
			}
		}
		for (size_t i1 = 0, iend1 = vMatched1.size(); i1 < iend1; i1++) {
			if (vMatched1[i1] >= 0) {
				int i2 = vMatched1[i1];
				vecMatches.push_back(std::make_pair(i1, i2));
			}
		}
		return vecMatches.size();
	}

	int ObjectMatcher::SearchFrameByProjection(BoundingBox* pBox, BoxFrame* pFrame, std::vector<std::pair<int, int>>& vecMatches, float thProjection, float thMaxDesc, float thMinDesc, bool bCheckOri)
	{
		int nmatches = 0;

		// Rotation Histogram (to check rotation consistency)
		std::vector<int> rotHist[HISTO_LENGTH];
		float factor = 1.0f / HISTO_LENGTH;
		cv::Mat Tcw = pFrame->GetPose();
		cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
		cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
		cv::Mat twc = -Rcw.t() * tcw;

		//std::cout << "11 = " << pBox->mvWorld.size() << " " << pBox->mvKPs.size() << " " << pBox->desc.rows << std::endl;
		//std::cout << pFrame->mnMaxX << " " << pFrame->cx << std::endl;
		for (int i = 0, N = pBox->mvWorld.size(); i < N; i++)
		{
			if (pBox->mvDepth[i] < 0.0)
				continue;
			// Project
			cv::Mat x3Dw = pBox->mvWorld[i];
			cv::Mat x3Dc = Rcw * x3Dw + tcw;

			float xc = x3Dc.at<float>(0);
			float yc = x3Dc.at<float>(1);
			float invzc = 1.0 / x3Dc.at<float>(2);

			if (invzc < 0)
				continue;

			float u = pFrame->fx * xc * invzc + pFrame->cx;
			float v = pFrame->fy * yc * invzc + pFrame->cy;

			if (u<pFrame->mnMinX || u>pFrame->mnMaxX)
				continue;
			if (v<pFrame->mnMinY || v>pFrame->mnMaxY)
				continue;

			int nLastOctave = pBox->mvKeyDatas[i].octave;
			// Search in a window. Size depends on scale
			float radius = thProjection * pFrame->mvScaleFactors[nLastOctave];
			
			std::vector<size_t> vIndices2;
			vIndices2 = pFrame->GetFeaturesInArea(u, v, radius);
			
			if (vIndices2.empty())
				continue;

			cv::Mat dMP = pBox->mDescriptors.row(i);

			int bestDist = 256;
			int bestIdx2 = -1;

			for (std::vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++)
			{
				size_t i2 = *vit;
				if (pFrame->mvbMatched[i2])
					continue;
				
				const cv::Mat& d = pFrame->mDescriptors.row(i2);

				int dist = (int)Utils::CalcBinaryDescriptor(dMP, d);
				
				if (dist < bestDist)
				{
					bestDist = dist;
					bestIdx2 = i2;
				}
			}

			if (bestDist <= thMaxDesc)
			{
				pFrame->mvbMatched[bestIdx2] = true;
				nmatches++;
				vecMatches.push_back(std::make_pair(i, bestIdx2));
			}
		}
		return nmatches;
	}
}