#include <GlobalInstance.h>

#include <ObjectSLAM.h>
#include <BoxFrame.h>
#include <FrameInstance.h>

#include <KeyFrame.h>
#include <MapPoint.h>
#include <Utils_Geometry.h>
#include <EIF.h>

namespace ObjectSLAM {
	std::atomic<long unsigned int> GlobalInstance::mnNextGIId = 0;
	ObjectSLAM* GlobalInstance::ObjSystem = nullptr;
	GlobalInstance::GlobalInstance() :mnId(++mnNextGIId), mnMatchFail(0), mbBad(false), pos(cv::Mat::zeros(3, 1, CV_32FC1)) {
	}

	void GlobalInstance::EIFFilterOutlier()
	{

		//unique_lock<mutex> lock(mMutexMapPoints);

		//Extended Isolation Forest
		std::vector<std::array<float, 3>> data;
		std::vector<int> vecIndex;

		auto vpMPs = this->AllMapPoints.ConvertVector();

		if (mapConnected.Size() < 4 || vpMPs.size() < 20) {
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

		double th = 0.5;//mfEIFthreshold;

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


		{
			std::stringstream ss;
			ss << "EIF::filter," << this->mnId << "," << this->mapConnected.Size() << "==" << this->AllMapPoints.Size() << "," << nErr << "," << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
			//ss << "G::NewMP," << this->mnId << "," << spMPs.size() << "," << nAdd;
			ObjSystem->vecObjectAssoRes.push_back(ss.str());
		}

	}

	void GlobalInstance::Connect(FrameInstance* pIns, BoxFrame* pBF, int id) {
		//bf에 인덱스 추가
		this->mapConnected.Update(pBF, id);
		this->mapInstances.Update(pIns, id);

		//ObjSystem->vecObjectAssoRes.push_back("G::C::start");

		//현재 포인트가 mask 를 벗어나는 경우 제거
		auto pKF = pBF->mpRefKF;
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

		{
			std::stringstream ss;
			ss << "G::DelMP," << this->mnId << "," << this->mapConnected.Size() << "," << this->AllMapPoints.Size() << ", " << "," << nDel;
			ObjSystem->vecObjectAssoRes.push_back(ss.str());
		}

		//추가 instance의 mp 추가
		//ObjSystem->vecObjectAssoRes.push_back("G::C::123");
		this->AddMapPoints(pIns->setMPs);
		//ObjSystem->vecObjectAssoRes.push_back("G::C::end");
	}

	void GlobalInstance::Merge(GlobalInstance* pG) {
		if (this->mnId == pG->mnId)
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
		}
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
		auto tempBFs = this->mapConnected.Get();
		auto tempMapInstances = this->mapInstances.Get();
		//ObjSystem->vecObjectAssoRes.push_back("G::M::start");
		std::map<FrameInstance*, cv::Mat> mapR, mapT, mapK;
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
		}
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

			for (auto pair : tempMapInstances)
			{
				auto p = pair.first;
				if (!mapR.count(p))
				{
					//ObjSystem->vecObjectAssoRes.push_back("G::M::err");
					continue;
				}
				cv::Mat tempR = mapR[p];
				cv::Mat tempT = mapT[p];
				cv::Mat tempK = mapK[p];

				auto rect = p->rect;

				float d = 0.0;
				cv::Point2f pt;
				bool bproj = CommonUtils::Geometry::ProjectPoint(pt, d, pMP->GetWorldPos(), tempK, tempR, tempT);
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
		{
			std::stringstream ss;
			ss << "G::NewMP," << this->mnId << "," << this->mapConnected.Size() << "," << this->AllMapPoints.Size() << "," << spMPs.size() << "," << nAdd << "," << nAlready;
			ObjSystem->vecObjectAssoRes.push_back(ss.str());
		}
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
			std::unique_lock<std::mutex> lock(mMutexPos);
			pos = cv::Mat::zeros(3, 1, CV_32FC1);
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
			std::unique_lock<std::mutex> lock(mMutexPos);
			pos = cv::Mat::zeros(3, 1, CV_32FC1);
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
			std::unique_lock<std::mutex> lock(mMutexPos);
			pos = pca.mean.t();
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
	}

	cv::Mat GlobalInstance::GetPosition() {
		std::unique_lock<std::mutex> lock(mMutexPos);
		return pos.clone();
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