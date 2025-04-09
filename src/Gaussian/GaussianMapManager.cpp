#include <Gaussian/GaussianMapManager.h>
#include <Gaussian/GaussianObject.h>

#include <FrameInstance.h>
#include <KeyFrame.h>

namespace ObjectSLAM {
	bool GaussianMapManager::InitializeObject(GOMAP::GaussianObject* pG) {
		auto obs = pG->GetObservations();
		if (obs.size() < 2){
			std::cout << "���� �� ����" << std::endl;
			return false;
		}

		float min_sim = 0.9998;

		FrameInstance* prev = nullptr;
		FrameInstance* curr = nullptr;

		for (auto pair1 : obs)
		{
			auto p1 = pair1.second;
			auto pKF1 = p1->mpRefKF;

			const float& fx1 = pKF1->fx;
			const float& fy1 = pKF1->fy;
			const float& cx1 = pKF1->cx;
			const float& cy1 = pKF1->cy;
			const float& invfx1 = pKF1->invfx;
			const float& invfy1 = pKF1->invfy;

			float x1 = p1->pt.x;
			float y1 = p1->pt.y;

			cv::Mat Rcw1 = pKF1->GetRotation();
			cv::Mat Rwc1 = Rcw1.t();

			cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (x1 - cx1) * invfx1, (y1 - cy1) * invfy1, 1.0);
			cv::Mat ray1 = Rwc1 * xn1;
			ray1 /= cv::norm(ray1);

			for (auto pair2 : obs)
			{
				auto p2 = pair2.second;
				if (p1 == p2)
					continue;
				auto pK2 = p2->mpRefKF;	
				auto pKF2 = p2->mpRefKF;

				const float& fx2 = pKF2->fx;
				const float& fy2 = pKF2->fy;
				const float& cx2 = pKF2->cx;
				const float& cy2 = pKF2->cy;
				const float& invfx2 = pKF2->invfx;
				const float& invfy2 = pKF2->invfy;

				float x2 = p2->pt.x;
				float y2 = p2->pt.y;
				
				cv::Mat Rcw2 = pKF2->GetRotation();
				cv::Mat Rwc2 = Rcw2.t();

				cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (x2 - cx2) * invfx2, (y2 - cy2) * invfy2, 1.0);
				cv::Mat ray2 = Rwc2 * xn2;
				ray2 /= cv::norm(ray2);

				const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

				if (cosParallaxRays < min_sim) {
					//���� �� �����ؾ� �� ���� ����.
					min_sim = cosParallaxRays;
					prev = p1;
					curr = p2;
				}
			}
		}
		
		if (!prev || !curr)
			return false;
		
		if (InitializeObject(pG, prev, curr)){
			//std::cout << "���� ����" << std::endl;
			return true;
		}
		/*else
			std::cout << "���� ����" << std::endl;*/
		return false;
	}

	bool GaussianMapManager::InitializeObject(GOMAP::GaussianObject* pG, FrameInstance* pPrev, FrameInstance* pCurr) {
		
		//�ʱ� ������ ȹ��
		auto pKF1 = pPrev->mpRefKF;
		auto pKF2 = pCurr->mpRefKF;

		const float& fx1 = pKF1->fx;
		const float& fy1 = pKF1->fy;
		const float& cx1 = pKF1->cx;
		const float& cy1 = pKF1->cy;
		const float& invfx1 = pKF1->invfx;
		const float& invfy1 = pKF1->invfy;

		const float& fx2 = pKF2->fx;
		const float& fy2 = pKF2->fy;
		const float& cx2 = pKF2->cx;
		const float& cy2 = pKF2->cy;
		const float& invfx2 = pKF2->invfx;
		const float& invfy2 = pKF2->invfy;

		cv::Mat Tcw1 = pKF1->GetPose();
		cv::Mat Tcw2 = pKF2->GetPose();

		float x1 = pPrev->pt.x;
		float y1 = pPrev->pt.y;

		float x2 = pCurr->pt.x;
		float y2 = pCurr->pt.y;

		cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (x1 - cx1) * invfx1, (y1 - cy1) * invfy1, 1.0);
		cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (x2 - cx2) * invfx2, (y2 - cy2) * invfy2, 1.0);

		cv::Mat Rcw1 = Tcw1.rowRange(0, 3).colRange(0, 3);
		cv::Mat Rcw2 = Tcw2.rowRange(0, 3).colRange(0, 3);

		cv::Mat Rwc1 = Rcw1.t();
		cv::Mat Rwc2 = Rcw2.t();

		cv::Mat tcw1 = Tcw1.rowRange(0, 3).col(3);
		cv::Mat tcw2 = Tcw2.rowRange(0, 3).col(3);

		cv::Mat ray1 = Rwc1 * xn1;
		cv::Mat ray2 = Rwc2 * xn2;

		/*ray1 /= cv::norm(ray1);
		ray2 /= cv::norm(ray2);*/

		const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

		if (cosParallaxRays > 0.9998) {

			double dotProduct = ray1.dot(ray2);
			dotProduct = std::max(-1.0, std::min(1.0, dotProduct));
			double angleRad = std::acos(dotProduct);
			double angleDeg = angleRad * 180.0 / CV_PI;

			std::cout << "GO::cos sim error = " << cosParallaxRays << " " << angleDeg << std::endl;
			return false;
		}
		//��� 3D ��ġ ���
		//SVD �̿�?

		//�ʱ� ��Ȯ�Ǽ� ���

		//�ٿ�� �ڽ�

		//�������̼� �߰�
		//cv::Mat ray1, ray2;
		cv::Mat mean;
		if (!GaussianMapManager::triangulatePoint(xn1, xn2, Tcw1, Tcw2, mean)) {
			//std::cout << "GO::�ﰢȭ ���� " << std::endl;
			return false;
		}
		if (!CheckObjectPosition(mean, pPrev->pt, Rcw1, tcw1, fx1, fy1, cx1, cy1)) {
			//std::cout << "GO::Frame1::����Ʈ ���� ����" << std::endl;
			return false;
		}
		if (!CheckObjectPosition(mean, pCurr->pt, Rcw2, tcw2, fx2, fy2, cx2, cy2))
		{
			//std::cout << "GO::Frame2::����Ʈ ���� ����" << std::endl;
			return false;
		}
		
		//���� ���л�
		cv::Mat Rwo = Tcw1.rowRange(0, 3).colRange(0, 3).t();
		cv::Mat cov1, cov2, cov;
		int n1 = GaussianMapManager::computeCovariance(cov1, pPrev, Tcw1, Rwo, mean, invfx1, invfy1);
		int n2 = GaussianMapManager::computeCovariance(cov2, pCurr, Tcw2, Rwo, mean, invfx2, invfy2);
		cov = cov1 + cov2;

		//auto pGO = new GOMAP::GaussianObject(mean, cov, Rwo);
		pG->Initialize(mean, cov, Rwo);
		pG->nContour = n1 + n2;
		//pG->nObs = 2;
		return true;
	}

	GOMAP::GaussianObject* GaussianMapManager::InitializeObject(FrameInstance* pPrev, FrameInstance* pCurr) {

		//�ʱ� ������ ȹ��
		auto pKF1 = pPrev->mpRefKF;
		auto pKF2 = pCurr->mpRefKF;

		const float& fx1 = pKF1->fx;
		const float& fy1 = pKF1->fy;
		const float& cx1 = pKF1->cx;
		const float& cy1 = pKF1->cy;
		const float& invfx1 = pKF1->invfx;
		const float& invfy1 = pKF1->invfy;

		const float& fx2 = pKF2->fx;
		const float& fy2 = pKF2->fy;
		const float& cx2 = pKF2->cx;
		const float& cy2 = pKF2->cy;
		const float& invfx2 = pKF2->invfx;
		const float& invfy2 = pKF2->invfy;

		cv::Mat Tcw1 = pKF1->GetPose();
		cv::Mat Tcw2 = pKF2->GetPose();

		float x1 = pPrev->pt.x;
		float y1 = pPrev->pt.y;

		float x2 = pCurr->pt.x;
		float y2 = pCurr->pt.y;

		cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (x1 - cx1) * invfx1, (y1 - cy1) * invfy1, 1.0);
		cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (x2 - cx2) * invfx2, (y2 - cy2) * invfy2, 1.0);
		
		cv::Mat Rcw1 = pKF1->GetRotation();
		cv::Mat Rcw2 = pKF2->GetRotation();

		cv::Mat Rwc1 = Rcw1.t();
		cv::Mat Rwc2 = Rcw2.t();

		cv::Mat tcw1 = pKF1->GetTranslation();
		cv::Mat tcw2 = pKF2->GetTranslation();

		cv::Mat ray1 = Rwc1 * xn1;
		cv::Mat ray2 = Rwc2 * xn2;

		ray1 /= cv::norm(ray1);
		ray2 /= cv::norm(ray2);

		const float cosParallaxRays = ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2));

		if (cosParallaxRays > 0.9998){

			double dotProduct = ray1.dot(ray2);
			dotProduct = std::max(-1.0, std::min(1.0, dotProduct));
			double angleRad = std::acos(dotProduct);
			double angleDeg = angleRad * 180.0 / CV_PI;

			std::cout << "GO::cos sim error = " << cosParallaxRays <<" "<<angleDeg << std::endl;
			return new GOMAP::GaussianObject();
		}
		//��� 3D ��ġ ���
		//SVD �̿�?

		//�ʱ� ��Ȯ�Ǽ� ���

		//�ٿ�� �ڽ�

		//�������̼� �߰�
		//cv::Mat ray1, ray2;
		cv::Mat mean;
		if (!GaussianMapManager::triangulatePoint(xn1, xn2, Tcw1, Tcw2, mean)){
			std::cout << "GO::�ﰢȭ ���� " << std::endl;
			return new GOMAP::GaussianObject();
		}
		if (!CheckObjectPosition(mean, pPrev->pt, Rcw1, tcw1, fx1, fy1, cx1, cy1)){
			std::cout << "GO::Frame1::����Ʈ ���� ����" << std::endl;
			return new GOMAP::GaussianObject();
		}
		if (!CheckObjectPosition(mean, pCurr->pt, Rcw2, tcw2, fx2, fy2, cx2, cy2))
		{
			std::cout << "GO::Frame2::����Ʈ ���� ����" << std::endl;
			return new GOMAP::GaussianObject();
		}
		//���Ͻ�, ���� ���� üũ

		//�������� ���
		/*auto Ow1 = cv::Point3f(pKF1->GetCameraCenter());
		auto Ow2 = cv::Point3f(pKF1->GetCameraCenter());
		auto ray1 = cv::Point3f(xn1);
		auto ray2 = cv::Point3f(xn2);
		cv::Point3f intersection;
		float distance;
		compute3DLineIntersection(Ow1, ray1, Ow2, ray2, intersection, distance);
		mean = cv::Mat(intersection);*/

		////EKF ���л�
		////covariance
		//cv::Mat R1, Ow1;
		//cv::Mat R2, Ow2;
		//cv::Rect rect1 = pPrev->rect;
		//cv::Rect rect2 = pCurr->rect;
		//cv::Mat cov1, cov2;
		//GaussianMapManager::computeCovariance(cov1, rect1, mean, Tcw1, invfx1, invfy1);
		//GaussianMapManager::computeCovariance(cov2, rect2, mean, Tcw2, invfx2, invfy2);

		////�ʱ� �ں����� ���
		////1. ���
		////cv::Mat cov = (cov1 + cov2) / 2.0;
		////2. �������� ���
		//cv::Mat cov = (cov1.inv() + cov2.inv()).inv();
		////EKF ���л�

		

		//���� ���л�
		cv::Mat Rwo = Tcw1.rowRange(0, 3).colRange(0, 3).t();
		cv::Mat cov1, cov2, cov;
		int n1 = GaussianMapManager::computeCovariance(cov1, pPrev, Tcw1, Rwo, mean, invfx1, invfy1);
		int n2 = GaussianMapManager::computeCovariance(cov2, pCurr, Tcw2, Rwo, mean, invfx2, invfy2);
		cov = cov1 + cov2;

		//��ü �� ����
		//auto pGaussianObj = std::make_unique<GaussianObject>(mean, cov);
		auto pGO = new GOMAP::GaussianObject(mean, cov, Rwo);
		pGO->nContour = n1 + n2;
		pGO->nObs = 2;
		return pGO;

	}
	void GaussianMapManager::UpdateObjectWithIncremental(GOMAP::GaussianObject* pGO, FrameInstance* pCurr) {
		auto pKF2 = pCurr->mpRefKF;

		const float& fx2 = pKF2->fx;
		const float& fy2 = pKF2->fy;
		const float& cx2 = pKF2->cx;
		const float& cy2 = pKF2->cy;
		const float& invfx2 = pKF2->invfx;
		const float& invfy2 = pKF2->invfy;

		cv::Mat Tcw2 = pKF2->GetPose();
		cv::Mat K2 = pKF2->K.clone();

		cv::Mat old_u = pGO->GetPosition();
		cv::Mat Rwo = pGO->Rwo.clone();
		cv::Mat old_cov = pGO->GetCovariance();
		//int old_n = pGO->nContour-1;

		//mean ����ȭ

		//cov update
		cv::Mat ncov;
		pGO->nContour += GaussianMapManager::computeCovariance(ncov, pCurr, Tcw2, Rwo, old_u, invfx2, invfy2);
		pGO->SetCovariance(ncov+old_cov);

		//������ �̿�ÿ� nContour-1 ������
		//�ð�ȭ �Ҷ� ���� ���� ������ ȸ����Ű��
	}
	void GaussianMapManager::UpdateObjectWithEKF(GOMAP::GaussianObject* pGO, FrameInstance* pCurr) {

		auto pKF2 = pCurr->mpRefKF;

		const float& fx2 = pKF2->fx;
		const float& fy2 = pKF2->fy;
		const float& cx2 = pKF2->cx;
		const float& cy2 = pKF2->cy;
		const float& invfx2 = pKF2->invfx;
		const float& invfy2 = pKF2->invfy;

		cv::Mat Tcw2 = pKF2->GetPose();
		cv::Mat K2 = pKF2->K.clone();

		cv::Mat u = pGO->GetPosition();
		cv::Mat cov = pGO->GetCovariance();

		//�߽�
		cv::Rect rect = pCurr->rect; //pCurr�� rect
		cv::Mat z = cv::Mat::zeros(2, 1, CV_32FC1);
		z.at<float>(0) = pCurr->pt.x;
		z.at<float>(1) = pCurr->pt.y;

		//projection
		cv::Mat Rcw2 = Tcw2.rowRange(0, 3).colRange(0, 3);
		cv::Mat tcw2 = Tcw2.rowRange(0, 3).col(3);
		cv::Mat p = Rcw2 * u + tcw2;
		cv::Mat tmp_h = K2 * p;
		cv::Mat h = cv::Mat::zeros(2,1,CV_32FC1);
		h.at<float>(0) = tmp_h.at<float>(0) / tmp_h.at<float>(2);
		h.at<float>(1) = tmp_h.at<float>(1) / tmp_h.at<float>(2);

		//jacobian
		cv::Mat H;
		computeJacobian(H, Rcw2, p, fx2, fy2);
		
		//compute innovation and kalman gain
		cv::Mat R = cv::Mat::eye(2, 2, CV_32FC1);
		R.at<float>(0, 0) = rect.width * rect.width * 0.25;
		R.at<float>(1, 1) = rect.height * rect.height * 0.25;

		/*cv::Mat R, Ra;
		GaussianMapManager::computeCovariance(Ra, rect, u, Tcw2, invfx2, invfy2);
		R = H * Ra * H.t();*/

		cv::Mat y = z - h;
		cv::Mat S = H * cov * H.t() + R;
		cv::Mat K = cov * H.t() * S.inv();

		/*pGO->mean = u + K * y;
		pGO->covariance = (cv::Mat::eye(3, 3, CV_32FC1) - K * H) * cov;*/
		pGO->nObs++;
		std::cout << "update test = " << pGO->nObs << std::endl;
		/// <summary>
		/// ���� ����
		/// </summary>
		/// <param name="pGO"></param>
		/// <param name="pCurr"></param>
		//cv::Mat ray; //contour�� �߽��� 3�������� ǥ����.

		//cv::Mat Ow = pKF2->GetCameraCenter(); //frame���� ī�޶� ��ġ
		//
		//cv::Mat measurement_cov;
		//float depth = cv::norm(pGO->mean - Ow);
		//GaussianMapManager::computeCovariance(measurement_cov, rect, pGO->mean, Tcw2, invfx2, invfy2);
		//cv::Mat K = pGO->covariance * (pGO->covariance + measurement_cov).inv();

		//cv::Mat predicted_ray = pGO->mean - Ow;
		//cv::normalize(predicted_ray, predicted_ray);
		//cv::Mat innovation = ray - predicted_ray;

		//cv::Mat new_mean = pGO->mean + K * innovation;
		//cv::Mat new_cov = (cv::Mat::eye(3, 3, CV_32F) - K) * pGO->covariance;
		//
		//pGO->mean= new_mean;
		//pGO->covariance = new_cov;

	}


	void GaussianMapManager::pointToRay(cv::Mat& ray,const cv::Point2f& pt, const cv::Mat& Rwc, float fx, float fy, float cx, float cy){
		ray = (cv::Mat_<float>(3, 1) <<
			(pt.x - cx) / fx,
			(pt.y - cy) / fy,
			1.0f
		);
		ray = Rwc * ray;
		cv::normalize(ray, ray);
	}

	//line_point : ī�޶� ����, dir�� ����
	bool GaussianMapManager::compute3DLineIntersection(
		const cv::Point3f& line1_point, const cv::Point3f& line1_direction,
		const cv::Point3f& line2_point, const cv::Point3f& line2_direction,
		cv::Point3f& intersection, float& distance) {

		// ���� ������ ����ȭ
		cv::Point3f d1 = line1_direction / cv::norm(line1_direction);
		cv::Point3f d2 = line2_direction / cv::norm(line2_direction);

		// �� ���� ������ ����
		cv::Point3f cross = d1.cross(d2);
		float cross_norm = cv::norm(cross);

		// �� ������ ������ ���
		if (cross_norm < 1e-6f) {
			// ù ��° �������� �� ��° ���������� ���� ���� ���
			cv::Point3f v = line2_point - line1_point;
			cv::Point3f proj = d1 * (v.dot(d1));

			// �Ÿ� ���
			cv::Point3f perpendicular = v - proj;
			distance = cv::norm(perpendicular);

			// ù ��° �������� ���� ����� �� ��ȯ
			intersection = line1_point;
			return false;
		}

		// �� ������ �������� ���� ���
		// ���� ����� ���� ����ϱ� ���� ������ ����
		cv::Point3f v = line2_point - line1_point;

		// �� ���� s�� t�� ����ϱ� ���� ��Ľ� �ذ�
		float a = d1.dot(d1);
		float b = d1.dot(d2);
		float c = d2.dot(d2);
		float d = d1.dot(v);
		float e = d2.dot(v);

		float denom = a * c - b * b;

		// �и� 0�̸� �� ������ �����ϰų� ��ġ
		if (std::abs(denom) < 1e-6f) {
			distance = 0.0f;
			intersection = line1_point;
			return false;
		}

		// ���� ����� ���� �ش��ϴ� �Ű����� �� ���
		float s = (b * e - c * d) / denom;
		float t = (a * e - b * d) / denom;

		// �� �������� ���� ����� �� ���
		cv::Point3f closest_point1 = line1_point + d1 * s;
		cv::Point3f closest_point2 = line2_point + d2 * t;

		// �� ���� ������ �ּ� �Ÿ� ���
		distance = cv::norm(closest_point2 - closest_point1);

		// ������ �Ǵ� �߰��� ����
		if (distance < 1e-6f) {
			intersection = closest_point1;
			return true; // ���� ����
		}
		else {
			// �� ������ �������� ������ �߰��� ��ȯ
			intersection = (closest_point1 + closest_point2) * 0.5f;
			return false;
		}
	}

	bool GaussianMapManager::CheckObjectPosition(const cv::Mat& X, cv::Point2f pt, const cv::Mat& R, const cv::Mat& t
		, float fx, float fy, float cx, float cy, float sigmaSquare) {
		
		cv::Mat Xt = X.t();

		float z1 = R.row(2).dot(Xt) + t.at<float>(2);
		
		if (z1 <= 0)
			return false;
		
		const float x1 = R.row(0).dot(Xt) + t.at<float>(0);
		const float y1 = R.row(1).dot(Xt) + t.at<float>(1);
		const float invz1 = 1.0 / z1;
		
		float u1 = fx * x1 * invz1 + cx;
		float v1 = fy * y1 * invz1 + cy;
		float errX1 = u1 - pt.x;
		float errY1 = v1 - pt.y;
		
		if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare)
			return false;
		
		return true;
	}

	bool GaussianMapManager::triangulatePoint(const cv::Mat& xn1, const cv::Mat& xn2, const cv::Mat& Tcw1, const cv::Mat& Tcw2, cv::Mat& x3D){
		//�ﰢȭ Ȯ��

		cv::Mat A(4, 4, CV_32F);
		A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
		A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
		A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
		A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

		cv::Mat w, u, vt;
		cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

		x3D = vt.row(3).t();

		if (x3D.at<float>(3) == 0)
			return false;

		// Euclidean coordinates
		x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
		return true;
	}

	int GaussianMapManager::computeCovariance(cv::Mat& cov, FrameInstance* pIns, 
		const cv::Mat& Tcw, const cv::Mat& Rwo,
		const cv::Mat& mean, float invfx, float invfy) 
	{
		//���� contour�� ���� return
		auto pt = pIns->pt;
		auto contours = pIns->contour;

		cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
		cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
		cv::Mat Ow = -Rcw.t() * tcw;

		const cv::Mat Rwc = Rcw.t();
		const cv::Mat Row = Rwo.t();

		float depth = cv::norm(mean - Ow);

		cov = cv::Mat::zeros(3, 3, CV_32FC1);

		for (auto c : contours)
		{
			auto diff = (cv::Point2f)c - pt;

			float x = diff.x*invfx;
			float y = diff.y*invfy;
			float z = (x + y) / 2;

			cov.at<float>(0, 0) += x * x;
			cov.at<float>(1, 1) += y * y;
			//cov.at<float>(2, 2) += z * z;

		}
		cov = depth*depth * Row *Rwc * cov * Rcw * Rwo;
		return contours.size();
	}

	void GaussianMapManager::computeCovariance(cv::Mat& cov, const cv::Rect& rect, const cv::Mat& mean, const cv::Mat& Tcw, float invfx, float invfy){
		cv::Mat R = Tcw.rowRange(0, 3).colRange(0, 3);
		cv::Mat t = Tcw.rowRange(0, 3).col(3);
		cv::Mat Ow = -R.t() * t;

		float depth = cv::norm(mean - Ow);

		float scale_x = rect.width * depth * invfx;
		float scale_y = rect.height * depth * invfy;
		float scale_z = (scale_x + scale_y) / 2;

		//����° ���� K.inv()*pt�� �ϴ��� ������ ������.

		cov = cv::Mat::zeros(3, 3, CV_32F);
		cov.at<float>(0, 0) = scale_x * scale_x * 0.25;
		cov.at<float>(1, 1) = scale_y * scale_y * 0.25;
		cov.at<float>(2, 2) = scale_z * scale_z * 0.25;

		//ī�޶� �ڼ��� ȸ����.
		cov = R.t() * cov * R;
	} 

	void GaussianMapManager::computeJacobian(cv::Mat& j, const cv::Mat& R, const cv::Mat& X, float fx, float fy) {
		/*cv::Mat R = T.rowRange(0, 3).colRange(0, 3);
		cv::Mat t = T.rowRange(0, 3).col(3);
		
		cv::Mat X = R* mean + t;*/

		float x = X.at<float>(0);
		float y = X.at<float>(1);
		float z = X.at<float>(2);

		j = cv::Mat::zeros(2, 3, CV_32FC1);
		j.at<float>(0, 0) = fx;
		j.at<float>(0, 2) = -x / z * fx;

		j.at<float>(1, 1) = fy;
		j.at<float>(1, 2) = -y / z * fy;

		j = -1 / z * j * R;
	}

}