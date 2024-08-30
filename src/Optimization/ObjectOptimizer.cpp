#include <Optimization/ObjectOptimizer.h>

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

namespace ObjectSLAM {
	int ObjectOptimizer::ObjectPoseOptimization(const std::vector<cv::Point2f>& imgPts, const std::vector<cv::Point3f>& objPts, std::vector<bool>& outliers, cv::Mat& P, const float fx, const float fy, const float cx, const float cy) {
		g2o::SparseOptimizer optimizer;
		auto linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
		auto block_solver = std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
		auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
		optimizer.setAlgorithm(algorithm);

		int nInitialCorrespondences = 0;

		// Set Frame vertex
		//P = cv::Mat::eye(4, 4, CV_32FC1);
		auto vSE3 = new BaseSLAM::Optimization::VertexFrame();
		vSE3->setEstimate(CommonUtils::DataConverter::toSE3Quat(P));
		vSE3->setId(0);
		vSE3->setFixed(false);
		optimizer.addVertex(vSE3);

		// Set MapPoint vertices
		int N = imgPts.size();

		std::vector<BaseSLAM::Optimization::EdgeSE3MonoFrame*> vpEdgesMono;
		std::vector<size_t> vnIndexEdgeMono;
		vpEdgesMono.reserve(N);
		vnIndexEdgeMono.reserve(N);

		float th = 5.991;
		float deltaMono = sqrt(th);

		float Na = 0;
		{
			
			for (int i = 0; i < N; i++) {

				auto imgpt = imgPts[i];
				auto objpt = objPts[i];
				nInitialCorrespondences++;
				
				Eigen::Matrix<double, 2, 1> obs;
				obs << imgpt.x, imgpt.y;

				auto e = new BaseSLAM::Optimization::EdgeSE3MonoFrame();

				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
				e->setMeasurement(obs);
				float invSigma2 = 1.0f;// pBox->mvInvLevelSigma2[kp.octave];
				e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

				g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
				e->setRobustKernel(rk);
				rk->setDelta(deltaMono);

				e->fx_ = fx;
				e->fy_ = fy;
				e->cx_ = cx;
				e->cy_ = cy;
				
				e->pos_w[0] = objpt.x;
				e->pos_w[1] = objpt.y;
				e->pos_w[2] = objpt.z;

				optimizer.addEdge(e);
				vpEdgesMono.push_back(e);
				vnIndexEdgeMono.push_back(i);
			}
		}

		if (nInitialCorrespondences < 3) {
			//delete linearSolver;
			//delete vSE3;
			return 0;
		}

		float chi2Mono[4] = { th,th,th,th };
		int its[4] = { 10,10,10,10 };

		int nBad = 0;
		for (size_t it = 0; it < 4; it++)
		{

			vSE3->setEstimate(CommonUtils::DataConverter::toSE3Quat(P));
			optimizer.initializeOptimization(0);
			optimizer.optimize(its[it]);

			nBad = 0;
			for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; i++)
			{
				auto e = vpEdgesMono[i];

				size_t idx = vnIndexEdgeMono[i];

				if (outliers[idx])
				{
					e->computeError();
				}

				float chi2 = e->chi2();

				if (chi2 > chi2Mono[it])
				{
					outliers[idx] = true;
					e->setLevel(1);
					nBad++;
				}
				else
				{
					outliers[idx] = false;
					e->setLevel(0);
				}

				if (it == 2)
					e->setRobustKernel(0);
			}
			if (nBad == nInitialCorrespondences)
				break;
			//std::cout << "Object pOse opti test = " << nBad << " " << nInitialCorrespondences << " " << optimizer.edges().size() << std::endl;
			if (optimizer.edges().size() < 10)
				break;
		}

		// Recover optimized pose and return number of inliers
		auto vSE3_recov = static_cast<BaseSLAM::Optimization::VertexFrame*>(optimizer.vertex(0));
		auto SE3quat_recov = vSE3_recov->estimate();
		cv::Mat pose = CommonUtils::DataConverter::toCvMat(SE3quat_recov);
		P = pose.clone();
		//std::cout << "Test obj pose = " << pose <<avgPos.t()<< std::endl;

		return nInitialCorrespondences - nBad;
	}
}