#include <Gaussian/Visualizer.h>
#include <Gaussian/GaussianObject.h>

namespace ObjectSLAM {
    
    void GaussianVisualizer::visualize2D(cv::Mat& image,
        GOMAP::GaussianObject* pGO,
        const cv::Mat& K,
        const cv::Mat& Rcw,
        const cv::Mat& tcw,
        const cv::Scalar& color,
        float confidence,
        int thickness) 
    {
        cv::Mat cov = pGO->GetCovariance();
        cov /= (pGO->nContour - 1);

        //원점 계산
        cv::Mat Xc = (Rcw * pGO->GetPosition() + tcw);
        cv::Mat Xi = K*Xc;
        cv::Point2f center(Xi.at<float>(0) / Xi.at<float>(2), Xi.at<float>(1) / Xi.at<float>(2));

        //타원 프로젝션
        
        cv::Mat Rwc = Rcw.t();
        cv::Mat Rwo = pGO->Rwo;
        cv::Mat Row = Rwo.t();

        cv::Mat cov2D;
        float fx = K.at<float>(0, 0);
        float fy = K.at<float>(1, 1);
        float x = Xc.at<float>(0);
        float y = Xc.at<float>(1);
        float invz = 1/Xc.at<float>(2);
        float invz2 = invz * invz;
        cv::Mat J = cv::Mat::zeros(2, 3, CV_32FC1);
        
        J.at<float>(0, 0) = fx * invz;
        J.at<float>(0, 2) = -fx *x* invz2;

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
        float angle = atan2(eigenvectors.at<float>(1, 0), eigenvectors.at<float>(0, 0));

        // 타원의 장축/단축 길이 계산
        float axes_length_major = sqrt(chi2_val * eigenvalues.at<float>(0, 0));
        float axes_length_minor = sqrt(chi2_val * eigenvalues.at<float>(1, 0));

        // OpenCV 타원 그리기 파라미터로 변환
        
        cv::Size axes(cvRound(axes_length_major), cvRound(axes_length_minor));
        double angle_deg = angle * 180.0 / CV_PI;
        
        // 타원 그리기
        cv::ellipse(image, center, axes, angle_deg, 0, 360, color, thickness);

        // 중심점 표시
        cv::putText(image, std::to_string(pGO->id), center, 2, 1.3, cv::Scalar(255, 0, 0), 2);
    }

    void GaussianVisualizer::visualize3D(cv::Mat& image,
        GOMAP::GaussianObject* pGO,
        const cv::Mat& K,
        const cv::Mat& R,
        const cv::Mat& t,
        const cv::Scalar& color,
        float scale)
    {
        
        //증분 공분산일 때
        cv::Mat cov = pGO->GetCovariance();
        cov /= (pGO->nContour - 1);
        //cov = (R * cov * R.t()) / (pGO->nContour - 1);

        // 3D 타원체 포인트 생성
        std::vector<cv::Point3f> points3d =
            generateEllipsoidPoints(cov, cv::Point3f(pGO->GetPosition()), scale);

        //std::cout << "go::vis::2 = " << points3d.size()<< std::endl;
        // 2D로 투영
        std::vector<cv::Point2f> points2d;
        //cv::projectPoints(points3d, R, t, K, cv::Mat(), points2d);

        for (size_t i = 0; i < points3d.size(); ++i) {
            cv::Mat point(points3d[i]);
            /*objectPoints.at<float>(i, 0) = corners[i].x;
            objectPoints.at<float>(i, 1) = corners[i].y;
            objectPoints.at<float>(i, 2) = corners[i].z;*/
            cv::Mat tmp = K * (R * point + t);
            float d = tmp.at<float>(2);
            points2d.push_back(cv::Point2f(tmp.at<float>(0) / d, tmp.at<float>(1) / d));
        }

        //std::cout << "go::vis::3" << std::endl;

        cv::Mat tmp = K * (R * pGO->GetPosition() + t);
        float d = tmp.at<float>(2);
        auto cpt = (cv::Point2f(tmp.at<float>(0) / d, tmp.at<float>(1) / d));
        cv::circle(image, cpt, 10, cv::Scalar(255, 0, 255), -1);

        // 타원체 윤곽선 그리기
        for (size_t i = 0; i < points2d.size() - 1; i++) {
            //std::cout << "go::vis::points::" << points2d[i] << std::endl;
            cv::line(image, points2d[i], points2d[i + 1], color, 3);
        }
        
    }

    std::vector<cv::Point3f> GaussianVisualizer::generateEllipsoidPoints(const cv::Mat& covariance,
        const cv::Point3f& center,
        float scale,
        int resolution)  
    {
        cv::Mat eigenvalues, eigenvectors;
        cv::eigen(covariance, eigenvalues, eigenvectors);

        std::vector<cv::Point3f> points;
        float phi_step = 2 * CV_PI / resolution;
        float theta_step = CV_PI / resolution;

        float x = cv::sqrt(eigenvalues.at<float>(0));
        float y = cv::sqrt(eigenvalues.at<float>(1));
        float z = cv::sqrt(eigenvalues.at<float>(2));
        cv::Mat tmp =  (cv::Mat_<float>(3, 1) << x, y, z);
        tmp = scale*eigenvectors*cv::Mat::diag(tmp);

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

                points.push_back(cv::Point3f(
                    transformed.at<float>(0) + center.x,
                    transformed.at<float>(1) + center.y,
                    transformed.at<float>(2) + center.z
                ));
            }
        }
        return points;
    }

}