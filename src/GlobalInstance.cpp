#include <GlobalInstance.h>

#include <ObjectSLAM.h>
#include <BoxFrame.h>
#include <FrameInstance.h>

#include <KeyFrame.h>
#include <MapPoint.h>
#include <Camera.h>
#include <Utils_Geometry.h>
#include <EIF.h>

#include <Gaussian/GaussianObject.h>
#include <Eval/EvalObj.h>

namespace ObjectSLAM {
	std::atomic<long unsigned int> GlobalInstance::mnNextGIId = 0;
	ObjectSLAM* GlobalInstance::ObjSystem = nullptr;
	GlobalInstance::GlobalInstance() :mnId(++mnNextGIId), mnMatchFail(0), mbBad(false), pos(cv::Mat::zeros(3, 1, CV_32FC1))
		, cov(cv::Mat::eye(3,3,CV_32FC1)), mpEval(nullptr){
	}

	void GlobalInstance::EIFFilterOutlier()
	{
		//unique_lock<mutex> lock(mMutexMapPoints);

		//Extended Isolation Forest
		std::vector<std::array<float, 3>> data;
		std::vector<int> vecIndex;

		auto vpMPs = this->AllMapPoints.ConvertVector();

		if (setConnected.Size() < 4 || vpMPs.size() < 20) {
			//std::cout << "asdf eif" <<" "<<mapConnected.Size()<<" "<<vpMPs.size() << std::endl;
			return;
		}
		for (int i = 0; i < vpMPs.size(); i++)
		{
			auto pMP = vpMPs[i];
			if (!pMP || pMP->isBad())
				continue;
			std::array<float, 3> temp;
			cv::Mat pos = pMP->GetWorldPos();
			for (uint32_t j = 0; j < 3; j++)
			{
				temp[j] = pos.at<float>(j);
			}
			data.push_back(temp);
			vecIndex.push_back(i);
		}

		auto t1 = std::chrono::system_clock::now();

		EIF::EIForest<float, 3> forest;

		double th = 0.6;//mfEIFthreshold;

		//Appropriately expand the EIF threshold for non-textured objects
		/*if (mnClass == 73 || mnClass == 46 || mnClass == 41)
		{
			th = th + 0.02;
		}*/

		double th_serious = th + 0.1;

		int point_num = 0;
		if (vpMPs.size() > 100)
			point_num = vpMPs.size() / 2;
		else
			point_num = vpMPs.size() * 2 / 3;

		if (!forest.Build(40, 12345, data, point_num))
		{
			std::cerr << "Failed to build Isolation Forest.\n";
		}

		std::vector<double> anomaly_scores;

		if (!forest.GetAnomalyScores(data, anomaly_scores))
		{
			std::cerr << "Failed to calculate anomaly scores.\n";
		}

		int nErr = 0;
		for (size_t i = 0, iend = vecIndex.size(); i < iend; i++) {
			int idx = vecIndex[i];
			auto pMPi = vpMPs[idx];
			if (anomaly_scores[i] > th)
			{
				this->AllMapPoints.Erase(pMPi);
				nErr++;
			}
			//std::cout << "eif = " << anomaly_scores[i] << std::endl;
		}

		//std::vector<EdgeSLAM::MapPoint*> newVpMapPoints;
		//for (size_t i = 0, iend = vpMPs.size(); i < iend; i++)
		//{
		//	MapPoint* pMP = mvpMapPoints[i];

		//	//outlier                   If the point is added for a long time, it is considered stable
		//	if (mnCheckMPsObs)
		//	{
		//		if (anomaly_scores[i] > th_serious)
		//		{
		//			pMP->EraseObject(this);
		//		}
		//		else if (anomaly_scores[i] > th && mnlatestObsFrameId - pMP->mAssociateObjects[this] < mnEIFObsNumbers)
		//		{
		//			pMP->EraseObject(this);
		//		}
		//		else
		//			newVpMapPoints.push_back(pMP);
		//	}
		//	else
		//	{
		//		if (anomaly_scores[i] > th)
		//		{
		//			pMP->EraseObject(this);
		//		}
		//		else
		//			newVpMapPoints.push_back(pMP);
		//	}

		//}

		//mvpMapPoints = newVpMapPoints;
		// 
		auto t2 = std::chrono::system_clock::now();

		//{
		//	std::stringstream ss;
		//	ss << "EIF::filter," << this->mnId << "," << this->mapConnected.Size() << "==" << this->AllMapPoints.Size() << "," << nErr << "," << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
		//	//ss << "G::NewMP," << this->mnId << "," << spMPs.size() << "," << nAdd;
		//	ObjSystem->vecObjectAssoRes.push_back(ss.str());
		//}

	}

	void GlobalInstance::GetLocalMPs(std::set<EdgeSLAM::MapPoint*>& spMPs, EdgeSLAM::KeyFrame* pKF, float angle, float dist, int bin) {
		//pPrevG의 인접 프레임 얻고 그 프레임의 맵포인트가 현재 프레임의 마스크에 있는가 체크
		cv::Mat dir = this->GetPosition() - pKF->GetCameraCenter();
		auto res = Utils::CalcSphericalCoordinate(dir, angle, dist); //0.1
		auto keykf = std::make_pair(res.x, res.y);
		std::vector<cv::Point> vecKeys = Utils::GetNeighborSphericalCoordinate(cv::Point(res.x, res.y), angle, bin);
		
		for (auto key : vecKeys)
		{
			auto tmp = std::make_pair(key.x, key.y);
			auto mapMPs = this->MapMPs.Get(tmp);
			for (auto pair : mapMPs)
			{
				auto setTempMP = pair.second;
				for (auto pMPi : setTempMP)
				{
					if (!pMPi || pMPi->isBad())
						continue;
					if (spMPs.count(pMPi))
						continue;
					spMPs.insert(pMPi);
				}
			}
		}
	}
	void GlobalInstance::GetProjPTs(const std::set<EdgeSLAM::MapPoint*>& spMPs, std::vector<cv::Point2f>& vecPTs, EdgeSLAM::KeyFrame* pKF) {
		
		const cv::Mat T = pKF->GetPose();
		const cv::Mat K = pKF->K.clone();
		const cv::Mat R = T.rowRange(0, 3).colRange(0, 3);
		const cv::Mat t = T.rowRange(0, 3).col(3);

		int w = pKF->mpCamera->mnWidth;
		int h = pKF->mpCamera->mnHeight;

		for (auto pMPi : spMPs)
		{
			if (!pMPi || pMPi->isBad())
				continue;
			float d;
			cv::Point2f pt;
			if (CommonUtils::Geometry::ProjectPoint(pt, d, pMPi->GetWorldPos(), K, R, t))
			{
				if (CommonUtils::Geometry::IsInImage(pt, w, h))
				{
					vecPTs.push_back(pt);
				}
			}
		}
	}

	cv::Point2f GetCenter(const std::vector<cv::Point2f>& points) {

		cv::Point2f res(0, 0);
		for (auto pt : points)
		{
			res += pt;
		}
		res.x /= points.size();
		res.y /= points.size();
		return res;
	}

	cv::Rect GlobalInstance::GetRect(const std::vector<cv::Point2f>& points) {
		if (points.size() < 2)
			return cv::Rect(0,0,0,0);

		float min_x = points[0].x;
		float min_y = points[0].y;
		float max_x = points[0].x;
		float max_y = points[0].y;

		for (const auto& pt : points) {
			min_x = std::min(min_x, pt.x);
			min_y = std::min(min_y, pt.y);
			max_x = std::max(max_x, pt.x);
			max_y = std::max(max_y, pt.y);
		}

		return cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y);
	}

	void GlobalInstance::Update(EdgeSLAM::KeyFrame* pKF) {

		std::vector<cv::Mat> mat;
		this->Update(mat);

		cv::Mat dir = this->GetPosition() - pKF->GetCameraCenter();
		auto res = Utils::CalcSphericalCoordinate(dir, 45.0, 0.01); //0.1
		std::set<EdgeSLAM::KeyFrame*> spKFs;
		auto keykf = std::make_pair(res.x, res.y);
		if (this->MapKFs.Count(keykf))
		{
			spKFs = this->MapKFs.Get(keykf);
		}
		spKFs.insert(pKF);
		this->MapKFs.Update(keykf, spKFs);
		//std::cout << pKF->mnId << " = spherical test = " << pPrevG->mnId << " " << res << " == " << spKFs.size() << std::endl;

		this->MapMPs.Clear();
		auto spMPs = this->AllMapPoints.Get();
		for (auto pMPi : spMPs)
		{
			if (!pMPi || pMPi->isBad())
				continue;
			cv::Mat dir2 = this->GetPosition() - pMPi->GetWorldPos();
			auto res2 = Utils::CalcSphericalCoordinate(dir2, 45.0, 0.01); //0.1
			auto keymp1 = std::make_pair(res2.x, res2.y);
			auto keymp2 = res2.z;
			std::map<int, std::set<EdgeSLAM::MapPoint*>> spMPs;
			if (this->MapMPs.Count(keymp1))
			{
				spMPs = this->MapMPs.Get(keymp1);
			}
			spMPs[keymp2].insert(pMPi);
			this->MapMPs.Update(keymp1, spMPs);
		}
	}

	void GlobalInstance::Connect(FrameInstance* pIns) {
		//bf에 인덱스 추가
		this->setConnected.Update(pIns);
		//this->mapConnected.Update(pBF, id);
		//this->mapInstances.Update(pIns, id);

		//ObjSystem->vecObjectAssoRes.push_back("G::C::start");

		//현재 포인트가 mask 를 벗어나는 경우 제거
		auto pKF = pIns->mpRefKF;
		cv::Mat T = pKF->GetPose();
		cv::Mat R = T.rowRange(0, 3).colRange(0, 3);
		cv::Mat t = T.rowRange(0, 3).col(3);
		cv::Mat K = pKF->K.clone();
		auto rect = pIns->rect;
		auto spMPs = this->AllMapPoints.Get();

		int nDel = 0;

		for (auto pMPi : spMPs)
		{
			if (!pMPi || pMPi->isBad())
			{
				this->AllMapPoints.Erase(pMPi);
				continue;
			}
			cv::Point2f pt;
			float d = 0.0;
			bool bproj = CommonUtils::Geometry::ProjectPoint(pt, d, pMPi->GetWorldPos(), K, R, t);
			
			if (!bproj || !rect.contains(pt))
			{
				this->AllMapPoints.Erase(pMPi);
				nDel++;
			}

			/*if (!rect.contains(pt))
			{
				AllMapPoints.Erase(pMPi);
			}*/
		}

		/*{
			std::stringstream ss;
			ss << "G::DelMP," << this->mnId << "," << this->mapConnected.Size() << "," << this->AllMapPoints.Size() << ", " << "," << nDel;
			ObjSystem->vecObjectAssoRes.push_back(ss.str());
		}*/
		
		//추가 instance의 mp 추가
		//ObjSystem->vecObjectAssoRes.push_back("G::C::123");
		this->AddMapPoints(pIns->setMPs);
		//ObjSystem->vecObjectAssoRes.push_back("G::C::end");
	}

	void GlobalInstance::Merge(GlobalInstance* pG) {
		/*if (this->mnId == pG->mnId)
			return;
		GlobalInstance* pG1, * pG2;
		if (pG->mapConnected.Size() > this->mapConnected.Size())
		{
			pG1 = pG;
			pG2 = this;
		}
		else {
			pG1 = this;
			pG2 = pG;
		}

		auto mapConnected = pG2->mapConnected.Get();
		for (auto pair : mapConnected)
		{
			auto pBF = pair.first;
			auto idx = pair.second;
			if (!pBF->mapMasks.Count("yoloseg"))
				continue;
			if (pG1->mapConnected.Count(pBF))
			{
				std::cout << "global instance merge :: error????" << std::endl;
			}
			pBF->mapMasks.Get("yoloseg")->MapInstances.Update(idx, pG1);
			pG1->mapConnected.Update(pBF, idx);
		}

		auto spMPs = pG2->AllMapPoints.Get();
		for (auto pMP : spMPs)
		{
			if (!pMP || pMP->isBad())
				continue;
			if (pG1->AllMapPoints.Count(pMP))
				continue;
			pG1->AllMapPoints.Update(pMP);
		}*/
	}

	void GlobalInstance::AddMapPoints(std::set<EdgeSLAM::MapPoint*> spMPs) {

		int n = this->AllMapPoints.Size();
		for (auto pMP : spMPs)
		{
			if (!pMP || pMP->isBad())
				continue;
			if (this->AllMapPoints.Count(pMP))
			{
				continue;
			}
			this->AllMapPoints.Update(pMP);
		}
		
		return;
		/*auto tempBFs = this->mapConnected.Get();
		auto tempMapInstances = this->mapInstances.Get();*/

		auto tempIns = this->setConnected.Get();
		//ObjSystem->vecObjectAssoRes.push_back("G::M::start");
		/*std::map<FrameInstance*, cv::Mat> mapR, mapT, mapK;
		for (auto pair : tempBFs)
		{
			auto pBF = pair.first;
			int sid = pair.second;
			auto pKF = pBF->mpRefKF;
			auto pIns = pBF->mapMasks.Get("yoloseg")->FrameInstances.Get(sid);
			if (!pIns)
			{
				std::cout << "error case as;dlfjas;ldfja;sldfjas;dlfj" << std::endl;
			}
			cv::Mat T = pKF->GetPose();
			cv::Mat K = pKF->K.clone();
			cv::Mat R = T.rowRange(0, 3).colRange(0, 3);
			cv::Mat t = T.rowRange(0, 3).col(3);
			mapR[pIns] = R;
			mapT[pIns] = t;
			mapK[pIns] = K;
		}*/

		//ObjSystem->vecObjectAssoRes.push_back("G::M::1");
		int nAdd = 0;
		int nAlready = 0;
		for (auto pMP : spMPs)
		{
			if (!pMP || pMP->isBad())
				continue;
			if (this->AllMapPoints.Count(pMP))
			{
				nAlready++;
				continue;
			}

			bool bAdd = true;

			for (auto p : tempIns)
			{
				//auto p = pair.first;
				//
				//if (!mapR.count(p))
				//{
				//	//ObjSystem->vecObjectAssoRes.push_back("G::M::err");
				//	continue;
				//}
				//cv::Mat tempR = mapR[p];
				//cv::Mat tempT = mapT[p];
				//cv::Mat tempK = mapK[p];

				auto rect = p->rect;
				auto pTempKF = p->mpRefKF;
				cv::Mat T = pTempKF->GetPose();
				cv::Mat K = pTempKF->K.clone();
				cv::Mat R = T.rowRange(0, 3).colRange(0, 3);
				cv::Mat t = T.rowRange(0, 3).col(3);

				float d = 0.0;
				cv::Point2f pt;
				bool bproj = CommonUtils::Geometry::ProjectPoint(pt, d, pMP->GetWorldPos(), K, R, t);
				if (!bproj || !rect.contains(pt))
				{
					bAdd = false;
					break;
				}
				/*if (!rect.contains(pt))
				{
					bAdd = false;
					break;
				}*/
			}

			if (bAdd) {
				nAdd++;
				this->AllMapPoints.Update(pMP);
			}
		}

		/*{
			std::stringstream ss;
			ss << "G::NewMP," << this->mnId << "," << this->mapConnected.Size() << "," << this->AllMapPoints.Size() << "," << spMPs.size() << "," << nAdd << "," << nAlready;
			ObjSystem->vecObjectAssoRes.push_back(ss.str());
		}*/

		//ObjSystem->vecObjectAssoRes.push_back("G::M::end");
		//std::cout << this->mnId << " add mp = " << this->AllMapPoints.Size() <<" || "<<n << std::endl;
	}

	cv::Point2f GlobalInstance::ProjectPoint(const cv::Mat T, const cv::Mat& K) {
		cv::Mat apos;
		{
			std::unique_lock<std::mutex> lock(mMutexPos);
			apos = pos.clone();
		}
		if (apos.at<float>(0) == 0.0 && apos.at<float>(1) == 0.0 && apos.at<float>(2) == 0.0)
			return cv::Point2f(-1, -1);
		cv::Mat R = T.rowRange(0, 3).colRange(0, 3);
		cv::Mat t = T.rowRange(0, 3).col(3);
		cv::Mat proj = K * (R * apos + t);
		float d = proj.at<float>(2);

		auto pt = cv::Point2f(-1, -1);
		if (d > 0) {
			pt.x = proj.at<float>(0) / d;
			pt.y = proj.at<float>(1) / d;
		}
		return pt;
	}

	//중점과 범위 내의 포인트 빼내기
	void GlobalInstance::Update(std::vector<cv::Mat>& mat, float val) {
		//ObjSystem->vecObjectAssoRes.push_back("g::update::start");
		auto vpMPs = AllMapPoints.ConvertVector();
		int n = 0;
		cv::Mat pointMat(0, 3, CV_32F);

		for (auto pMP : vpMPs) {
			if (!pMP || pMP->isBad())
				continue;
			/*if (pMP->Observations() < 3)
				continue;*/
			const cv::Mat X3d = pMP->GetWorldPos();  
			pointMat.push_back(X3d.t());
			n++;
		}

		//바운딩 박스 계산하면서 평균 위치 계산
		if (n < 3)
		{
			{
				std::unique_lock<std::mutex> lock(mMutexPos);
				pos = cv::Mat::zeros(3, 1, CV_32FC1);
				cov = cv::Mat::eye(3, 3, CV_32FC1);
			}
			{
				std::unique_lock<std::mutex> lock(mMutexBB);
				vecCorners.clear();
			}
			//ObjSystem->vecObjectAssoRes.push_back("g::update::end");
			return;
		}

		auto testMean = cv::mean(pointMat);

		cv::PCA pca(pointMat, cv::Mat(), cv::PCA::DATA_AS_ROW);

		cv::Mat eigenVectors = pca.eigenvectors;
		cv::Mat eigenValues = pca.eigenvalues;
		if (eigenVectors.rows != 3 || eigenVectors.cols != 3)
		{
			//ObjSystem->vecObjectAssoRes.push_back("g::update::end");
			{
				std::unique_lock<std::mutex> lock(mMutexPos);
				pos = cv::Mat::zeros(3, 1, CV_32FC1);
				cov = cv::Mat::eye(3, 3, CV_32FC1);
			}
			{
				std::unique_lock<std::mutex> lock(mMutexBB);
				vecCorners.clear();
			}
			return;
		}

		bool bDeter = cv::determinant(eigenVectors) < 0;
		if (bDeter) {
			eigenVectors.row(2) = -eigenVectors.row(2);
		}

		// Transform points to the principal component space
		cv::Mat transformedPoints = (pointMat - cv::repeat(pca.mean, pointMat.rows, 1)) * eigenVectors.t();
		// Compute min and max in the transformed space
		cv::Point3d minCoords, maxCoords;

		//ObjSystem->vecObjectAssoRes.push_back("update::6");

		//xyz 평균으로 계산하기
		cv::Scalar mean1, stddev1;
		cv::Scalar mean2, stddev2;
		cv::Scalar mean3, stddev3;
		cv::Mat absTransformed = (transformedPoints); //abs
		cv::meanStdDev(absTransformed.col(0), mean1, stddev1);
		cv::meanStdDev(absTransformed.col(1), mean2, stddev2);
		cv::meanStdDev(absTransformed.col(2), mean3, stddev3);

		float maxx = val * stddev1.val[0];
		float maxy = val * stddev2.val[0];
		float maxz = val * stddev3.val[0];
		{
			for (int i = 0; i < transformedPoints.rows; i++) {
				cv::Mat temp = transformedPoints.row(i);

				/*float ux = abs(temp.at<float>(0));
				float uy = abs(temp.at<float>(1));
				float uz = abs(temp.at<float>(2));*/
				float ux = abs((temp.at<float>(0) - mean1.val[0]));
				float uy = abs((temp.at<float>(1) - mean2.val[0]));
				float uz = abs((temp.at<float>(2) - mean3.val[0]));

				/*if (ux > maxx || uy > maxy || uz > maxz)
					continue;*/

				cv::Mat transformedCorner = temp * eigenVectors + pca.mean;	//1x3
				mat.push_back(transformedCorner.t());
			}
		}
		{
			cv::Mat D = cv::Mat::zeros(3, 3, CV_32FC1);
			for (int i = 0; i < 3; i++) {
				D.at<float>(i, i) = eigenValues.at<float>(i);
			}
			
			std::unique_lock<std::mutex> lock(mMutexPos);
			pos = pca.mean.t();
			cov = eigenVectors.t() * D * eigenVectors;
		}

		//boundingbox update
		maxCoords.x = mean1.val[0] + maxx;
		maxCoords.y = mean2.val[0] + maxy;
		maxCoords.z = mean3.val[0] + maxz;

		minCoords.x = mean1.val[0] - maxx;
		minCoords.y = mean2.val[0] - maxy;
		minCoords.z = mean3.val[0] - maxz;

		std::vector<cv::Point3f> acorners = {
			cv::Point3f(minCoords.x, minCoords.y, minCoords.z),
			cv::Point3f(maxCoords.x, minCoords.y, minCoords.z),
			cv::Point3f(maxCoords.x, maxCoords.y, minCoords.z),
			cv::Point3f(minCoords.x, maxCoords.y, minCoords.z),
			cv::Point3f(minCoords.x, minCoords.y, maxCoords.z),
			cv::Point3f(maxCoords.x, minCoords.y, maxCoords.z),
			cv::Point3f(maxCoords.x, maxCoords.y, maxCoords.z),
			cv::Point3f(minCoords.x, maxCoords.y, maxCoords.z)
		};

		//Update bounding box corners
		{
			std::unique_lock<std::mutex> lock(mMutexBB);
			vecCorners.clear();
			for (const auto& corner : acorners) {
				cv::Mat cornerMat = (cv::Mat_<float>(1, 3) << corner.x, corner.y, corner.z);
				cv::Mat transformedCorner = cornerMat * eigenVectors + pca.mean;
				vecCorners.push_back(cv::Point3f(transformedCorner));
			}
		}
		//this->EIFFilterOutlier();
	}

	cv::Mat GlobalInstance::GetPosition() {
		std::unique_lock<std::mutex> lock(mMutexPos);
		return pos.clone();
	}
	cv::Mat GlobalInstance::GetCovariance() {
		std::unique_lock<std::mutex> lock(mMutexPos);
		return cov.clone();
	}
	GOMAP::GO2D GlobalInstance::Project2D(const cv::Mat& K, const cv::Mat& Rcw, const cv::Mat& tcw) {

		cv::Mat acov, apos;
		{
			std::unique_lock<std::mutex> lock(mMutexPos);
			acov = cov.clone();
			apos = pos.clone();
		}

		//원점 계산
		cv::Mat Xc = (Rcw * apos + tcw);
		cv::Mat Xi = K * Xc;
		cv::Mat mu = cv::Mat::zeros(2, 1, CV_32FC1);
		mu.at<float>(0) = Xi.at<float>(0) / Xi.at<float>(2);
		mu.at<float>(1) = Xi.at<float>(1) / Xi.at<float>(2);
		//cv::Point2f center(Xi.at<float>(0) / Xi.at<float>(2), Xi.at<float>(1) / Xi.at<float>(2));

		//타원 프로젝션
		cv::Mat cov2D;
		float fx = K.at<float>(0, 0);
		float fy = K.at<float>(1, 1);
		float x = Xc.at<float>(0);
		float y = Xc.at<float>(1);
		float invz = 1 / Xc.at<float>(2);
		float invz2 = invz * invz;
		cv::Mat J = cv::Mat::zeros(2, 3, CV_32FC1);

		J.at<float>(0, 0) = fx * invz;
		J.at<float>(0, 2) = -fx * x * invz2;

		J.at<float>(1, 1) = fy * invz;
		J.at<float>(1, 2) = -fy * y * invz2;

		cov2D = J * cov * J.t();

		//시각화 코드
		/*std::map<float, float> chi2_dict = {
		{0.90f, 4.605f},
		{0.95f, 5.991f},
		{0.99f, 9.210f}
		};
		float chi2_val = chi2_dict.count(confidence) ? chi2_dict[confidence] : 5.991f;*/
		float chi2_val = 1.0;

		// 고유값 분해
		cv::Mat eigenvalues, eigenvectors;
		cv::eigen(cov2D, eigenvalues, eigenvectors);

		// 타원의 각도 계산 (라디안)
		float angle_rad = atan2(eigenvectors.at<float>(1, 0), eigenvectors.at<float>(0, 0));

		// 타원의 장축/단축 길이 계산
		float axes_length_major = sqrt(chi2_val * eigenvalues.at<float>(0, 0));
		float axes_length_minor = sqrt(chi2_val * eigenvalues.at<float>(1, 0));

		// OpenCV 타원 그리기 파라미터로 변환

		cv::Size axes(cvRound(axes_length_major), cvRound(axes_length_minor));
		double angle_deg = angle_rad * 180.0 / CV_PI;

		//라디안 이용
		GOMAP::GO2D g(mu, cov2D, axes_length_major, axes_length_minor, angle_rad);
		/*g.center = mu.clone();
		g.cov2D = cov2D.clone();
		g.major = axes_length_major;
		g.minor = axes_length_minor;
		g.angle_rad = angle_rad;*/
		return g;
	}
	void GlobalInstance::UpdatePosition() {
		auto vpMPs = AllMapPoints.ConvertVector();
		int n = 0;
		cv::Mat avgPos = cv::Mat::zeros(3, 1, CV_32FC1);

		for (auto pMP : vpMPs) {
			if (!pMP || pMP->isBad())
				continue;

			avgPos += pMP->GetWorldPos();
			n++;
		}

		if (n == 0)
			n++;
		avgPos /= n;

		{
			std::unique_lock<std::mutex> lock(mMutexPos);
			pos = avgPos.clone();
		}
	}
	void GlobalInstance::CalculateBoundingBox() {
		auto vpMPs = AllMapPoints.ConvertVector();
		int n = 0;
		cv::Mat avgPos = cv::Mat::zeros(3, 1, CV_32FC1);
		cv::Mat pointMat(0, 3, CV_32F);

		//std::cout <<this->mnId<< " == compute bb = " <<" "<<this->mapConnected.Size()<<" " << AllMapPoints.Size() << " " << vpMPs.size() << " " << n << std::endl;

		for (auto pMP : vpMPs) {
			if (!pMP || pMP->isBad())
				continue;

			const cv::Mat X3d = pMP->GetWorldPos();
			avgPos += X3d;
			pointMat.push_back(X3d.t());
			n++;
		}

		//바운딩 박스 계산하면서 평균 위치 계산
		if (n == 0)
		{
			n++;
			avgPos /= n;
			std::unique_lock<std::mutex> lock(mMutexPos);
			pos = avgPos.clone();
			return;
		}

		cv::PCA pca(pointMat, cv::Mat(), cv::PCA::DATA_AS_ROW);
		//std::cout << "OBB::2" << std::endl;

		// Get the principal components
		cv::Mat eigenVectors = pca.eigenvectors;
		cv::Mat eigenValues = pca.eigenvalues;

		// Ensure right-handed coordinate system
		if (cv::determinant(eigenVectors) < 0) {
			eigenVectors.row(2) = -eigenVectors.row(2);
		}
		//std::cout << "OBB::3" << std::endl;

		// Transform points to the principal component space
		cv::Mat transformedPoints = (pointMat - cv::repeat(pca.mean, pointMat.rows, 1)) * eigenVectors.t();
		// Compute min and max in the transformed space
		cv::Point3d minCoords, maxCoords;

		//xyz 평균으로 계산하기
		cv::Scalar mean1, stddev1;
		cv::Scalar mean2, stddev2;
		cv::Scalar mean3, stddev3;
		cv::Mat absTransformed = cv::abs(transformedPoints);
		cv::meanStdDev(absTransformed.col(0), mean1, stddev1);
		cv::meanStdDev(absTransformed.col(1), mean2, stddev2);
		cv::meanStdDev(absTransformed.col(2), mean3, stddev3);

		float val = 1.285;
		maxCoords.x = mean1.val[0] + val * stddev1.val[0];
		maxCoords.y = mean2.val[0] + val * stddev2.val[0];
		maxCoords.z = mean3.val[0] + val * stddev3.val[0];

		/*cv::Point3d avgAxis(0.0, 0.0, 0.0);
		for (int i = 0; i < transformedPoints.rows; i++) {
			float x = std::abs(transformedPoints.at<float>(i, 0));
			float y = std::abs(transformedPoints.at<float>(i, 1));
			float z = std::abs(transformedPoints.at<float>(i, 2));
			maxCoords += cv::Point3d(x, y, z);
		}
		maxCoords /= n;
		*/
		minCoords = -maxCoords;

		/*for (int i = 0; i < 3; ++i) {
			cv::minMaxLoc(transformedPoints.col(i), &minCoords.x + i, &maxCoords.x + i);
		}*/
		//std::cout << "OBB::4" << std::endl;
		// Compute OBB properties
		//center = cv::Point3f(pca.mean);

		std::vector<cv::Point3f> acorners = {
			cv::Point3f(minCoords.x, minCoords.y, minCoords.z),
			cv::Point3f(maxCoords.x, minCoords.y, minCoords.z),
			cv::Point3f(maxCoords.x, maxCoords.y, minCoords.z),
			cv::Point3f(minCoords.x, maxCoords.y, minCoords.z),
			cv::Point3f(minCoords.x, minCoords.y, maxCoords.z),
			cv::Point3f(maxCoords.x, minCoords.y, maxCoords.z),
			cv::Point3f(maxCoords.x, maxCoords.y, maxCoords.z),
			cv::Point3f(minCoords.x, maxCoords.y, maxCoords.z)
		};

		//Update bounding box corners
		{
			std::unique_lock<std::mutex> lock(mMutexPos);
			pos = pca.mean.t();
		}
		{
			std::unique_lock<std::mutex> lock(mMutexBB);
			vecCorners.clear();
			for (const auto& corner : acorners) {
				cv::Mat cornerMat = (cv::Mat_<float>(1, 3) << corner.x, corner.y, corner.z);
				cv::Mat transformedCorner = cornerMat * eigenVectors + pca.mean;
				vecCorners.push_back(cv::Point3f(transformedCorner));
			}
			//std::cout << "OBB::5 ==" << vecCorners.size() << std::endl;
		}
	}
	void GlobalInstance::ProjectBB(std::vector<cv::Point2f>& vecProjPoints, const cv::Mat& K, const cv::Mat& T) {
		cv::Mat R = T(cv::Rect(0, 0, 3, 3));
		cv::Mat t = T(cv::Rect(3, 0, 1, 3));
		//std::cout << "OBB::proj::1" << std::endl;

		std::vector<cv::Point3f> corners;
		{
			std::unique_lock<std::mutex> lock(mMutexBB);
			corners = vecCorners;
		}

		//실제 이용은 cv::Mat임.
		// Convert OBB corners to cv::Mat
		for (auto corner : corners) {
			cv::Mat point = (cv::Mat_<float>(3, 1) << corner.x, corner.y, corner.z);
			cv::Mat tmp = K * (R * point + t);
			float d = tmp.at<float>(2);
			auto pt = cv::Point2f(tmp.at<float>(0) / d, tmp.at<float>(1) / d);
			vecProjPoints.push_back(pt);
		}
		//for (size_t i = 0; i < corners.size(); ++i) {
		//	cv::Mat point(corners[i]);
		//	//std::cout << pt << std::endl;
		//}

		//std::cout << "OBB::proj::2" << std::endl;
		//// Project 3D points to 2D image plane
		//cv::projectPoints(objectPoints, R, t, K, D, imagePoints);
		//std::cout << "OBB::proj::3" << std::endl;
	}
	void GlobalInstance::DrawBB(cv::Mat& image, const std::vector<cv::Point2f>& projectedCorners) {
		std::vector<std::vector<int>> connections = {
		{0, 1}, {1, 2}, {2, 3}, {3, 0}, // Bottom face
		{4, 5}, {5, 6}, {6, 7}, {7, 4}, // Top face
		{0, 4}, {1, 5}, {2, 6}, {3, 7}  // Connecting edges
		};

		// Draw the edges
		if (projectedCorners.size() != 8) {
			return;
		}

		for (const auto& connection : connections) {
			cv::line(image, projectedCorners[connection[0]], projectedCorners[connection[1]],
				cv::Scalar(255, 255, 0), 2);
		}

		// Draw the corners
		for (const auto& corner : projectedCorners) {
			cv::circle(image, corner, 3, cv::Scalar(0, 0, 255), -1);
		}
	}
}