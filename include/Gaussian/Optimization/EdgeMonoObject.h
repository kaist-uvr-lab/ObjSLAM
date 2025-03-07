#ifndef OBJECT_SLAM_GAUSSIANMAP_EDGE_MONO_OBJECT2_H
#define OBJECT_SLAM_GAUSSIANMAP_EDGE_MONO_OBJECT2_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include "g2o/core/base_unary_edge.h"

#include <Optimize/se3quat.h>
#include <Optimize/VertexPoint.h>

namespace ObjectSLAM {
	namespace GOMAP{
		namespace Optimizer {
			class EdgeMonoObject : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, BaseSLAM::Optimization::VertexPoint>
            {
			public:
                EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

                EdgeMonoObject();
            	
                virtual bool read(std::istream& is);
                virtual bool write(std::ostream& os) const;

                // jacobian matrix (not mandotary to code by hand)
                void linearizeOplus() override;

                // reprojection error
                void computeError();

                bool isDepthPositive();

                Eigen::Vector2d cam_project(const Eigen::Vector3d& pos_c) const;

                //프레임의 카메라 자세에서 SEQuat을 미리 정의해야함. 
                BaseSLAM::Optimization::SE3Quat T_;
                double fx_, fy_, cx_, cy_;

			};
		}
	}
}

#endif