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

		GaussianObject::GaussianObject(const cv::Mat& _pos, const cv::Mat& _cov, const cv::Mat& _R)
			:mean(_pos), covariance(_cov), Rwo(_R), nObs(0), nContour(0), nSeg(0), id(++GaussianObject::mnNextId){
		}
        cv::RotatedRect GO2D::CalcEllipse(float chisq)
        {
            float x = this->center.at<float>(0);
            float y = this->center.at<float>(1);

            float major_axis = this->major * chisq;
            float minor_axis = this->minor * chisq;
            
            return cv::RotatedRect(cv::Point(x, y),cv::Size(cvRound(major_axis), cvRound(minor_axis)), this->angle_deg);
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

		void GaussianObject::AddObservation(InstanceMask* f, FrameInstance* obs, bool bType) {
            mObservations.Update(f, obs);
            if (bType)
                nSeg++;
            nObs++;
		}
		std::map<InstanceMask*, FrameInstance*> GaussianObject::GetObservations() {
			return mObservations.Get();
		}
        FrameInstance* GaussianObject::GetObservation(InstanceMask* f){
            return mObservations.Get(f);
        }
        float GaussianObject::CalcDistance3D(GaussianObject* other){
            cv::Mat diff = this->mean - other->mean;
            cv::Mat combined = this->covariance + other->covariance;
            
            cv::Matx33d invCov; 
            cv::invert(combined, invCov, cv::DECOMP_SVD); // SVD 분해를 통한 안정적인 역행렬 계산

            // 4. 마할라노비스 거리 계산: d^T * invCov * d
            float dist = cv::Mat(diff.t() * invCov * diff).at<float>(0);

            return std::sqrt(dist);
        }
	}

}