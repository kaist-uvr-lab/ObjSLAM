#include <OrientedBoundingBox.h>

namespace ObjectSLAM {

    OrientedBoundingBox::OrientedBoundingBox() {}
    OrientedBoundingBox::~OrientedBoundingBox(){}

    //입력 데이터를 미리 추가할 수도 있음.
    void OrientedBoundingBox::calculateOBB(const std::vector<cv::Point3f>& points) {

        // Convert points to Mat
        cv::Mat pointMat(points.size(), 3, CV_32F);
        for (size_t i = 0; i < points.size(); ++i) {
            pointMat.at<float>(i, 0) = points[i].x;
            pointMat.at<float>(i, 1) = points[i].y;
            pointMat.at<float>(i, 2) = points[i].z;
        }
        
        std::cout << "OBB::1" << std::endl;

        // Perform PCA
        cv::PCA pca(pointMat, cv::Mat(), cv::PCA::DATA_AS_ROW);
        std::cout << "OBB::2" << std::endl;
        // Get the principal components
        cv::Mat eigenVectors = pca.eigenvectors;
        cv::Mat eigenValues = pca.eigenvalues;

        // Ensure right-handed coordinate system
        if (cv::determinant(eigenVectors) < 0) {
            eigenVectors.row(2) = -eigenVectors.row(2);
        }
        std::cout << "OBB::3" << std::endl;
        // Transform points to the principal component space
        cv::Mat transformedPoints = (pointMat - cv::repeat(pca.mean, pointMat.rows, 1)) * eigenVectors.t();

        // Compute min and max in the transformed space
        cv::Point3d minCoords, maxCoords;
        /*float minx, maxx;
        float miny, maxy;
        float minz, maxz;
        cv::minMaxLoc()
        cv::minMaxLoc(transformedPoints.col(0), &minx, &maxx);*/
        for (int i = 0; i < 3; ++i) {
            cv::minMaxLoc(transformedPoints.col(i), &minCoords.x + i, &maxCoords.x + i);
        }
        std::cout << "OBB::4" << std::endl;
        // Compute OBB properties
        center = cv::Point3f(pca.mean);
        //dimensions = maxCoords - minCoords;
        //axes = eigenVectors;

        // Compute corners
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

        // Transform corners back to original space
        corners.clear();
        for (const auto& corner : acorners) {
            cv::Mat cornerMat = (cv::Mat_<float>(1, 3) << corner.x, corner.y, corner.z);
            cv::Mat transformedCorner = cornerMat * eigenVectors + pca.mean;
            corners.push_back(cv::Point3f(transformedCorner));
        }
    }

    void OrientedBoundingBox::projectOBBToImage(const cv::Mat& K, const cv::Mat& D, const cv::Mat& T) {

        imagePoints.clear();
        cv::Mat R = T(cv::Rect(0, 0, 3, 3));
        cv::Mat t = T(cv::Rect(3, 0, 1, 3));
        std::cout << "OBB::proj::1" << std::endl;
        // Convert OBB corners to cv::Mat
        cv::Mat objectPoints(corners.size(), 3, CV_32F);
        for (size_t i = 0; i < corners.size(); ++i) {
            cv::Mat point(corners[i]);
            /*objectPoints.at<float>(i, 0) = corners[i].x;
            objectPoints.at<float>(i, 1) = corners[i].y;
            objectPoints.at<float>(i, 2) = corners[i].z;*/
            cv::Mat tmp = K* (R * point + t);
            float d = tmp.at<float>(2);
            imagePoints.push_back(cv::Point2f(tmp.at<float>(0) / d, tmp.at<float>(1) / d));
        }
        //std::cout << "OBB::proj::2" << std::endl;
        //// Project 3D points to 2D image plane
        //cv::projectPoints(objectPoints, R, t, K, D, imagePoints);
        std::cout << "OBB::proj::3" << std::endl;
    }
    void OrientedBoundingBox::drawProjectedOBB(cv::Mat& image, const std::vector<cv::Point2f>& projectedCorners) {
        std::vector<std::vector<int>> connections = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, // Bottom face
        {4, 5}, {5, 6}, {6, 7}, {7, 4}, // Top face
        {0, 4}, {1, 5}, {2, 6}, {3, 7}  // Connecting edges
        };

        // Draw the edges
        for (const auto& connection : connections) {
            cv::line(image, projectedCorners[connection[0]], projectedCorners[connection[1]],
                cv::Scalar(0, 255, 0), 2);
        }

        // Draw the corners
        for (const auto& corner : projectedCorners) {
            cv::circle(image, corner, 3, cv::Scalar(0, 0, 255), -1);
        }
    }
}

