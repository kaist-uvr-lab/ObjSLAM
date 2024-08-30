#include <ObjectMatcher.h>
#include <Utils.h>

namespace ObjectSLAM {

	const int ObjectMatcher::HISTO_LENGTH = 30;

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