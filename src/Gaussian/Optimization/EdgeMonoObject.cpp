#include <Gaussian/Optimization/EdgeMonoObject.h>

namespace ObjectSLAM {
	namespace GOMAP{
		namespace Optimizer {

            EdgeMonoObject::EdgeMonoObject() : g2o::BaseUnaryEdge<2, Eigen::Vector2d, BaseSLAM::Optimization::VertexPoint>()
            {

            }
            bool EdgeMonoObject::read(std::istream& is) {
                g2o::internal::readVector(is, _measurement);
                return readInformationMatrix(is);
            }

            bool EdgeMonoObject::write(std::ostream& os) const {
                g2o::internal::writeVector(os, measurement());
                return writeInformationMatrix(os);
            }
            void EdgeMonoObject::linearizeOplus()
            {
                auto vi = static_cast<BaseSLAM::Optimization::VertexPoint*>(_vertices.at(0));           // data::keyframe
                Eigen::Vector3d pos_w = vi->BaseSLAM::Optimization::VertexPoint::estimate();
                const BaseSLAM::Optimization::SE3Quat& cam_pose_cw = T_;
                const Eigen::Matrix3d rot_cw = cam_pose_cw.rotation().toRotationMatrix(); // rotation matrix

                Eigen::Vector3d pos_c;
                pos_c = cam_pose_cw.map(pos_w);

                const auto x = pos_c(0);
                const auto y = pos_c(1);
                const auto z = pos_c(2);
                const auto z_sq = z * z;

                _jacobianOplusXi(0, 0) = -fx_ * rot_cw(0, 0) / z + fx_ * x * rot_cw(2, 0) / z_sq;
                _jacobianOplusXi(0, 1) = -fx_ * rot_cw(0, 1) / z + fx_ * x * rot_cw(2, 1) / z_sq;
                _jacobianOplusXi(0, 2) = -fx_ * rot_cw(0, 2) / z + fx_ * x * rot_cw(2, 2) / z_sq;

                _jacobianOplusXi(1, 0) = -fy_ * rot_cw(1, 0) / z + fy_ * y * rot_cw(2, 0) / z_sq;
                _jacobianOplusXi(1, 1) = -fy_ * rot_cw(1, 1) / z + fy_ * y * rot_cw(2, 1) / z_sq;
                _jacobianOplusXi(1, 2) = -fy_ * rot_cw(1, 2) / z + fy_ * y * rot_cw(2, 2) / z_sq;
            }

            void EdgeMonoObject::computeError()
            {
                const auto v1 = static_cast<const BaseSLAM::Optimization::VertexPoint*>(_vertices.at(0));
                Eigen::Vector3d pos_w = v1->estimate();
                const Eigen::Vector2d obs(_measurement);
                _error = obs - cam_project(T_.map(pos_w));
            }

            bool EdgeMonoObject::isDepthPositive()
            {
                const auto v1 = static_cast<const BaseSLAM::Optimization::VertexPoint*>(_vertices.at(0));
                Eigen::Vector3d pos_w = v1->estimate();
                return 0.0 < (T_.map(pos_w))(2);
            }

            Eigen::Vector2d EdgeMonoObject::cam_project(const Eigen::Vector3d& pos_c) const
            {
                // project 3D point to 2D image pixel coordinates
                return { fx_ * pos_c(0) / pos_c(2) + cx_, fy_ * pos_c(1) / pos_c(2) + cy_ };
            }

		}
	}
}