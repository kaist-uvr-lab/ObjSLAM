#include <Gaussian/GaussianObject.h>
#include <FrameInstance.h>
#include <BoxFrame.h>

namespace ObjectSLAM {
	/*GaussianObject::GaussianObject(const cv::Mat& pos, const cv::Matx33d& cov,
		const cv::Mat& feat, const cv::Rect2d& box,
		double obsNoise)
		: mean(pos), covariance(cov), features(feat.clone()),
		bbox(box), observationNoise(obsNoise) {}*/
	namespace GOMAP {

        std::atomic<long unsigned int> GaussianObject::mnNextId = 0;
        GaussianObject::GaussianObject() :mbInitialized(false), mpReplaced(nullptr)
            , nObs(0), nContour(0), nSeg(0), id(++GaussianObject::mnNextId) {}
		GaussianObject::GaussianObject(const cv::Mat& _pos, const cv::Mat& _cov, const cv::Mat& _R)
			:mean(_pos), covariance(_cov), Rwo(_R), nObs(0), nContour(0), nSeg(0), id(++GaussianObject::mnNextId)
        , mbInitialized(true), mpEval(nullptr), mpBaseObj(nullptr), mpReplaced(nullptr) {
		}
        cv::RotatedRect GO2D::CalcEllipse(float chisq)
        {
            //rotated rect는 major의 지름으로 표현이 되기 때문에 반지름으로 그리려면 2배를 함.
            float major_axis = this->major * chisq * 2;
            float minor_axis = this->minor * chisq * 2;
            return cv::RotatedRect(cv::Point(this->center),cv::Size(cvRound(major_axis), cvRound(minor_axis)), this->angle_deg);
        }
        cv::Rect GO2D::CalcRect(float chisq) {
            float x = this->center.at<float>(0);
            float y = this->center.at<float>(1);
            float major_axis = this->major * chisq;
            float minor_axis = this->minor * chisq;

            // 회전 각도를 라디안으로 변환
            float theta = this->angle_rad;
            float cos_theta = std::cos(theta); 
            float sin_theta = std::sin(theta);

            // 회전된 타원의 bounding box 크기 계산
            float width = 2 * std::sqrt(
                std::pow(major_axis * cos_theta, 2) +
                std::pow(minor_axis * sin_theta, 2)
            );
            float height = 2 * std::sqrt(
                std::pow(major_axis * sin_theta, 2) +
                std::pow(minor_axis * cos_theta, 2)
            );

            // bounding rectangle 좌상단 점 계산
            float x1 = x - width / 2;
            float y1 = y - height / 2;

            auto rrect = cv::Rect(
                static_cast<int>(x1),
                static_cast<int>(y1),
                static_cast<int>(width),
                static_cast<int>(height)
            );
            return rrect;
        }

        void GaussianObject::Initialize(const cv::Mat& _pos, const cv::Mat& _cov, const cv::Mat& _R) {
            {
                std::unique_lock<std::mutex> lock(mMutex);
                mean = _pos.clone();
                covariance = _cov.clone();
                Rwo = _R.clone();
            }
            mbInitialized = true;
        }

        void GaussianObject::SetPosition(const cv::Mat& _pos) {
            std::unique_lock<std::mutex> lock(mMutex);
            mean = _pos.clone();
        }
        cv::Mat GaussianObject::GetPosition() {
            std::unique_lock<std::mutex> lock(mMutex);
            return mean.clone();
        }
        void GaussianObject::SetCovariance(const cv::Mat& _cov){
            std::unique_lock<std::mutex> lock(mMutex);
            covariance = _cov.clone();
        }
        cv::Mat GaussianObject::GetCovariance() {
            std::unique_lock<std::mutex> lock(mMutex);
            return covariance.clone();
        }

        void GaussianObject::GenerateEllipsoidPoints(
            cv::Mat& points
            ,float scale,
            int resolution)
        {
            cv::Mat center = this->GetPosition();
            cv::Mat cov = this->GetCovariance();
            cov /= (this->nContour - 1);

            cv::Mat eigenvalues, eigenvectors;
            cv::eigen(cov, eigenvalues, eigenvectors);

            float phi_step = 2 * CV_PI / resolution;
            float theta_step = CV_PI / resolution;

            float x = cv::sqrt(eigenvalues.at<float>(0));
            float y = cv::sqrt(eigenvalues.at<float>(1));
            float z = cv::sqrt(eigenvalues.at<float>(2));

            points = cv::Mat::zeros(0, 3, CV_32FC1);
            cv::Mat tmp = (cv::Mat_<float>(3, 1) << x, y, z);
            tmp = scale * eigenvectors * cv::Mat::diag(tmp);

            for (int i = 0; i <= resolution; i++) {
                float theta = i * theta_step;
                for (int j = 0; j <= resolution; j++) {
                    float phi = j * phi_step;

                    // 단위 구 위의 점
                    cv::Mat p = (cv::Mat_<float>(3, 1) <<
                        sin(theta) * cos(phi),
                        sin(theta) * sin(phi),
                        cos(theta));

                    // 타원체로 변환
                    /*cv::Mat transformed = eigenvectors *
                        cv::Mat::diag(cv::sqrt(eigenvalues)) *
                        p * scale;*/
                    cv::Mat transformed = tmp * p;
                    transformed += center;
                    points.push_back(transformed.t());

                }
            }
        }

        GO2D GaussianObject::Project2D(const cv::Mat& K, const cv::Mat& Rcw, const cv::Mat& tcw) {

            cv::Mat cov = this->covariance.clone();
            cov /= (this->nContour - 1);

            //원점 계산
            cv::Mat Xc = (Rcw * this->mean + tcw);
            cv::Mat Xi = K * Xc;
            cv::Mat mu = cv::Mat::zeros(2, 1, CV_32FC1);
            mu.at<float>(0) = Xi.at<float>(0) / Xi.at<float>(2);
            mu.at<float>(1) = Xi.at<float>(1) / Xi.at<float>(2);
            //cv::Point2f center(Xi.at<float>(0) / Xi.at<float>(2), Xi.at<float>(1) / Xi.at<float>(2));

            //타원 프로젝션

            cv::Mat Rwc = Rcw.t();
            cv::Mat Rwo = this->Rwo;
            cv::Mat Row = Rwo.t();

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

            cov2D = J * Rcw * Rwo * cov * Row * Rwc * J.t();

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
            GO2D g(mu, cov2D, axes_length_major, axes_length_minor, angle_rad);
            /*g.center = mu.clone();
            g.cov2D = cov2D.clone();
            g.major = axes_length_major;
            g.minor = axes_length_minor;
            g.angle_rad = angle_rad;*/
            return g;
		}

        void GaussianObject::Merge(GaussianObject* pOther) {
            if (this->id == pOther->id){
                //std::cout << "같은 객체" << std::endl;
                return;
            }
            if (pOther->mpReplaced == this || this->mpReplaced == pOther)
            {
                //std::cout << "이미 결합" << std::endl;
                return;
            }
            GaussianObject* pG1, * pG2;
            if(pOther->id < this->id)
            //if (pOther->mObservations.Size() > this->mObservations.Size())
            {
                pG1 = pOther;
                pG2 = this;
            }
            else {
                pG1 = this;
                pG2 = pOther;
            }
            
            auto obs = pG2->mObservations.Get();
            pG2->mObservations.Clear();
            

            for (auto pair : obs)
            {
                auto pMask = pair.first;
                auto pid = pair.second;

                pMask->GaussianMaps.Update(pid, pG1);
                pG1->AddObservation(pMask, pid);
            }

            cv::Mat cov = pG1->GetCovariance() + pG2->GetCovariance();
            int nContour = pG1->nContour + pG2->nContour;
            pG1->SetCovariance(cov);
            pG1->nContour = nContour;
            
            pG2->mpReplaced = pG1;
        }

		void GaussianObject::AddObservation(InstanceMask* f, int id, bool bType) {
            mObservations.Update(f, id);
            if (bType)
                nSeg++;
            nObs++;
		}
		std::map<InstanceMask*, int> GaussianObject::GetObservations() {
			return mObservations.Get();
		}
        int GaussianObject::GetObservation(InstanceMask* f){
            return mObservations.Get(f);
        }
        float GaussianObject::CalcDistance3D(GaussianObject* other){
            
            cv::Mat diff = this->GetPosition() - other->GetPosition();
            cv::Mat combined = this->GetCovariance()+ other->GetCovariance();
            float n = this->nContour + other->nContour - 2;
            combined /= n;
            
            cv::Mat invCov; 
            cv::invert(combined, invCov, cv::DECOMP_SVD); // SVD 분해를 통한 안정적인 역행렬 계산
            
            // 4. 마할라노비스 거리 계산: d^T * invCov * d
            
            cv::Mat res = diff.t() * invCov * diff;
            float dist = res.at<float>(0);
            
            return std::sqrt(dist);
        }
        bool GaussianObject::IsSameObject(GaussianObject* pOther, float th) {
            float dist = this->CalcDistance3D(pOther);
            return dist < th;
        }
	}

}