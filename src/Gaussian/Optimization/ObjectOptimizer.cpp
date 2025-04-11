#include <Gaussian/Optimization/ObjectOptimizer.h>

#include "g2o/core/base_unary_edge.h"
#include <g2o/core/base_edge.h>
#include "g2o/core/base_binary_edge.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/core/block_solver.h"
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/robust_kernel_impl.h"

#include <Optimize/VertexPoint.h>
#include <Optimize/VertexFrame.h>
#include <Optimize/EdgeSE3MonoFrame.h>
#include <Optimize/EdgeSE3MonoFramePoint.h>
#include <Optimize/EdgeSE3StereoFrame.h>
#include <Optimize/EdgeSE3StereoFramePoint.h>

#include <Converter.h>
#include <FrameInstance.h>
#include <BoxFrame.h>
#include <KeyFrame.h>
#include <Gaussian/Optimization/EdgeMonoObject.h>

#include <Utils.h>
#include <Camera.h>

namespace ObjectSLAM {
	namespace GOMAP {
		namespace Optimizer {
			void ObjectOptimizer::ObjectPosOptimization(GaussianObject* pG) {

				int ntrial = 10;

				//63이 맞나? 흠...
				g2o::SparseOptimizer optimizer;
				auto linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
				auto block_solver = std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
				auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
				optimizer.setAlgorithm(algorithm);

				//오브젝트의 위치 설정
				auto vPoint = new BaseSLAM::Optimization::VertexPoint();
				
				const cv::Mat pos = pG->GetPosition();

				vPoint->setEstimate(EdgeSLAM::Converter::toVector3d(pos));
				vPoint->setId(0);
				vPoint->setFixed(false);
				optimizer.addVertex(vPoint);

				auto setFrames = pG->GetObservations();
				std::vector<EdgeMonoObject*> vpEdgesMono;

				//자유도 체크
				float th = 5.991;
				float deltaMono = sqrt(th);

				int nInitialCorrespondences = setFrames.size();

				for (auto pair : setFrames)
				{
					auto pMask = pair.first;
					auto pid = pair.second;
					auto pFrame = pMask->FrameInstances.Get(pid);
					auto pKF = pFrame->mpRefKF;
					auto rect = pFrame->rect;
					auto pt = pFrame->pt;

					Utils::undistortPoint(pt, pt, pKF->K, pKF->mpCamera->D);

					Eigen::Matrix<double, 2, 1> obs;
					obs << pt.x, pt.y;

					//edge 생성
					auto e = new EdgeMonoObject();
					e->setVertex(0, dynamic_cast<BaseSLAM::Optimization::VertexPoint*>(optimizer.vertex(0)));
					e->setMeasurement(obs);
					Eigen::Matrix2d i = Eigen::Matrix2d::Identity();
					i(0, 0) = 2.0 / rect.width;// / 2.0;
					i(1, 1) = 2.0 / rect.height;// / 2.0;
					e->setInformation(i);

					//로버스트 에스티메이션
					g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
					e->setRobustKernel(rk);
					rk->setDelta(deltaMono);

					//기타 정보
					e->fx_ = pKF->fx;
					e->fy_ = pKF->fy;
					e->cx_ = pKF->cx;
					e->cy_ = pKF->cy;
					e->T_ = CommonUtils::DataConverter::toSE3Quat(pKF->GetPose());

					//데이터 저장
					optimizer.addEdge(e);
					vpEdgesMono.push_back(e);
				}

				if (nInitialCorrespondences < 2)
					return;

				//최적화
				float chi2Mono[4] = { th,th,th,th };
				int its[4] = { ntrial,ntrial,ntrial,ntrial };

				int nBad = 0;
				for (size_t it = 0; it < 1; it++)
				{
					//아웃라이어 관리 안함
					vPoint->setEstimate(EdgeSLAM::Converter::toVector3d(pos));
					optimizer.initializeOptimization(0);
					optimizer.optimize(its[it]);
				}

				//업데이트
				pG->SetPosition(EdgeSLAM::Converter::toCvMat(vPoint->estimate()));

			}//object optimization
		}//namespace
	}
}
