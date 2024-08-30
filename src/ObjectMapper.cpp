#include <ObjectMapper.h>

#include <BoundingBox.h>
#include <SegInstance.h>

namespace ObjectSLAM {

	int ObjectMapper::TwoViewTriangulation(SegInstance* pB1, SegInstance* pB2, const std::vector<std::pair<int, int>>& vMatches, std::vector<std::pair<bool, cv::Mat>>& vecTriangulated) {
		const float& fx1 = pB1->fx;
		const float& fy1 = pB1->fy;
		const float& cx1 = pB1->cx;
		const float& cy1 = pB1->cy;
		const float& invfx1 = pB1->invfx;
		const float& invfy1 = pB1->invfy;

		const float& fx2 = pB2->fx;
		const float& fy2 = pB2->fy;
		const float& cx2 = pB2->cx;
		const float& cy2 = pB2->cy;
		const float& invfx2 = pB2->invfx;
		const float& invfy2 = pB2->invfy;

		cv::Mat Tcw1 = pB1->GetPose();
		cv::Mat Rcw1 = Tcw1.rowRange(0, 3).colRange(0, 3);
		cv::Mat Rwc1 = Rcw1.t();
		cv::Mat tcw1 = Tcw1.rowRange(0, 3).col(3);
		cv::Mat Ow1 = pB1->GetCenter();

		cv::Mat Tcw2 = pB2->GetPose();
		cv::Mat Rcw2 = Tcw2.rowRange(0, 3).colRange(0, 3);
		cv::Mat Rwc2 = Rcw2.t();
		cv::Mat tcw2 = Tcw2.rowRange(0, 3).col(3);
		cv::Mat Ow2 = pB2->GetCenter();

		const float ratioFactor = 1.5f * pB1->mfScaleFactor;

		int nres = 0;
		vecTriangulated = std::vector<std::pair<bool, cv::Mat>>(vMatches.size(), std::make_pair(false, cv::Mat()));
		for (int i = 0, N = vMatches.size(); i < N; i++) {
			//vecTriangulated[i].first = false;
			const int& idx1 = vMatches[i].first;
			const int& idx2 = vMatches[i].second;

			auto pOP1 = pB1->mvpMapDatas.get(idx1);
			if (pOP1)
				continue;

			const cv::KeyPoint& kp1 = pB1->mvKeyDataUns[idx1];
			const cv::KeyPoint& kp2 = pB2->mvKeyDataUns[idx2];
			bool bStereo1 = pB1->mvuRight[idx1] >= 0;
			bool bStereo2 = pB2->mvuRight[idx2] >= 0;

			cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
			cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

			cv::Mat ray1 = Rwc1 * xn1;
			cv::Mat ray2 = Rwc2 * xn2;
			const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

			float cosParallaxStereo = cosParallaxRays + 1;
			float cosParallaxStereo1 = cosParallaxStereo;
			float cosParallaxStereo2 = cosParallaxStereo;

			if (bStereo1)
				cosParallaxStereo1 = cos(2 * atan2(pB1->mb / 2, pB1->mvDepth[idx1]));
			if (bStereo2)
				cosParallaxStereo2 = cos(2 * atan2(pB2->mb / 2, pB2->mvDepth[idx2]));
			cosParallaxStereo = std::min(cosParallaxStereo1, cosParallaxStereo2);

			cv::Mat x3D;
			//if(cosParallaxRays > 0 && cosParallaxRays < 0.9998)
			if (cosParallaxRays < cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays < 0.9998))
			{
				// Linear Triangulation Method
				cv::Mat A(4, 4, CV_32F);
				A.row(0) = (xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0));
				A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
				A.row(2) = (xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0));
				A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

				cv::Mat w, u, vt;
				cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

				x3D = vt.row(3).t();

				if (x3D.at<float>(3) == 0)
					continue;

				// Euclidean coordinates
				x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);

			}
			else
				continue; //No stereo and very low parallax

			cv::Mat x3Dt = x3D.t();

			//Check triangulation in front of cameras
			float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
			if (z1 <= 0)
				continue;

			float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
			if (z2 <= 0)
				continue;

			//Check reprojection error in first keyframe
			const float& sigmaSquare1 = pB1->mvLevelSigma2[kp1.octave];
			const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
			const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
			const float invz1 = 1.0 / z1;

			float u1 = fx1 * x1 * invz1 + cx1;
			float v1 = fy1 * y1 * invz1 + cy1;
			float errX1 = u1 - kp1.pt.x;
			float errY1 = v1 - kp1.pt.y;
			if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
				continue;

			//Check reprojection error in second keyframe
			const float sigmaSquare2 = pB2->mvLevelSigma2[kp2.octave];
			const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
			const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
			const float invz2 = 1.0 / z2;
			float u2 = fx2 * x2 * invz2 + cx2;
			float v2 = fy2 * y2 * invz2 + cy2;
			float errX2 = u2 - kp2.pt.x;
			float errY2 = v2 - kp2.pt.y;
			if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
				continue;

			//Check scale consistency
			cv::Mat normal1 = x3D - Ow1;
			float dist1 = cv::norm(normal1);

			cv::Mat normal2 = x3D - Ow2;
			float dist2 = cv::norm(normal2);

			if (dist1 == 0 || dist2 == 0)
				continue;

			const float ratioDist = dist2 / dist1;
			const float ratioOctave = pB1->mvScaleFactors[kp1.octave] / pB2->mvScaleFactors[kp2.octave];

			/*if(fabs(ratioDist-ratioOctave)>ratioFactor)
			continue;*/
			if (ratioDist * ratioFactor<ratioOctave || ratioDist>ratioOctave * ratioFactor)
				continue;
			vecTriangulated[i].first = true;
			vecTriangulated[i].second = x3D.clone();
			nres++;
		}
		return nres;
	}

	int ObjectMapper::TwoViewTriangulation(BoundingBox* pB1, BoundingBox* pB2, const std::vector<std::pair<int, int>>& vMatches, std::vector<std::pair<bool, cv::Mat>>& vecTriangulated) {

		const float& fx1 = pB1->fx;
		const float& fy1 = pB1->fy;
		const float& cx1 = pB1->cx;
		const float& cy1 = pB1->cy;
		const float& invfx1 = pB1->invfx;
		const float& invfy1 = pB1->invfy;

		const float& fx2 = pB2->fx;
		const float& fy2 = pB2->fy;
		const float& cx2 = pB2->cx;
		const float& cy2 = pB2->cy;
		const float& invfx2 = pB2->invfx;
		const float& invfy2 = pB2->invfy;

		cv::Mat Tcw1 = pB1->GetPose();
		cv::Mat Rcw1 = Tcw1.rowRange(0, 3).colRange(0, 3);
		cv::Mat Rwc1 = Rcw1.t();
		cv::Mat tcw1 = Tcw1.rowRange(0, 3).col(3);
		cv::Mat Ow1 = pB1->GetCenter();

		cv::Mat Tcw2 = pB2->GetPose();
		cv::Mat Rcw2 = Tcw2.rowRange(0, 3).colRange(0, 3);
		cv::Mat Rwc2 = Rcw2.t();
		cv::Mat tcw2 = Tcw2.rowRange(0, 3).col(3);
		cv::Mat Ow2 = pB2->GetCenter();

		const float ratioFactor = 1.5f * pB1->mfScaleFactor;

		int nres = 0;
		vecTriangulated = std::vector<std::pair<bool, cv::Mat>>(vMatches.size(), std::make_pair(false, cv::Mat()));
		for (int i = 0, N = vMatches.size(); i < N; i++) {
			//vecTriangulated[i].first = false;
			const int& idx1 = vMatches[i].first;
			const int& idx2 = vMatches[i].second;

			auto pOP1 = pB1->mvpMapDatas.get(idx1);
			if (pOP1)
				continue;

			const cv::KeyPoint& kp1 = pB1->mvKeyDataUns[idx1];
			const cv::KeyPoint& kp2 = pB2->mvKeyDataUns[idx2];
			bool bStereo1 = pB1->mvuRight[idx1] >= 0;
			bool bStereo2 = pB2->mvuRight[idx2] >= 0;

			cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
			cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.0);

			cv::Mat ray1 = Rwc1 * xn1;
			cv::Mat ray2 = Rwc2 * xn2;
			const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

			float cosParallaxStereo = cosParallaxRays + 1;
			float cosParallaxStereo1 = cosParallaxStereo;
			float cosParallaxStereo2 = cosParallaxStereo;

			if (bStereo1)
				cosParallaxStereo1 = cos(2 * atan2(pB1->mb / 2, pB1->mvDepth[idx1]));
			if (bStereo2)
				cosParallaxStereo2 = cos(2 * atan2(pB2->mb / 2, pB2->mvDepth[idx2]));
			cosParallaxStereo = std::min(cosParallaxStereo1, cosParallaxStereo2);

			cv::Mat x3D;
			//if(cosParallaxRays > 0 && cosParallaxRays < 0.9998)
			if (cosParallaxRays < cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays < 0.9998))
			{
				// Linear Triangulation Method
				cv::Mat A(4, 4, CV_32F);
				A.row(0) = (xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0));
				A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
				A.row(2) = (xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0));
				A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

				cv::Mat w, u, vt;
				cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

				x3D = vt.row(3).t();

				if (x3D.at<float>(3) == 0)
					continue;

				// Euclidean coordinates
				x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);

			}
			else
				continue; //No stereo and very low parallax

			cv::Mat x3Dt = x3D.t();

			//Check triangulation in front of cameras
			float z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
			if (z1 <= 0)
				continue;

			float z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
			if (z2 <= 0)
				continue;

			//Check reprojection error in first keyframe
			const float& sigmaSquare1 = pB1->mvLevelSigma2[kp1.octave];
			const float x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
			const float y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
			const float invz1 = 1.0 / z1;

			float u1 = fx1 * x1 * invz1 + cx1;
			float v1 = fy1 * y1 * invz1 + cy1;
			float errX1 = u1 - kp1.pt.x;
			float errY1 = v1 - kp1.pt.y;
			if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
				continue;

			//Check reprojection error in second keyframe
			const float sigmaSquare2 = pB2->mvLevelSigma2[kp2.octave];
			const float x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
			const float y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
			const float invz2 = 1.0 / z2;
			float u2 = fx2 * x2 * invz2 + cx2;
			float v2 = fy2 * y2 * invz2 + cy2;
			float errX2 = u2 - kp2.pt.x;
			float errY2 = v2 - kp2.pt.y;
			if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
				continue;

			//Check scale consistency
			cv::Mat normal1 = x3D - Ow1;
			float dist1 = cv::norm(normal1);

			cv::Mat normal2 = x3D - Ow2;
			float dist2 = cv::norm(normal2);

			if (dist1 == 0 || dist2 == 0)
				continue;

			const float ratioDist = dist2 / dist1;
			const float ratioOctave = pB1->mvScaleFactors[kp1.octave] / pB2->mvScaleFactors[kp2.octave];

			/*if(fabs(ratioDist-ratioOctave)>ratioFactor)
			continue;*/
			if (ratioDist * ratioFactor<ratioOctave || ratioDist>ratioOctave * ratioFactor)
				continue;
			vecTriangulated[i].first = true;
			vecTriangulated[i].second = x3D.clone();
			nres++;
		}
		return nres;

	}
}