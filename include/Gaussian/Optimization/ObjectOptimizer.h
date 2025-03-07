#ifndef OBJECT_SLAM_GAUSSIANMAP_OPTIMIZER_H
#define OBJECT_SLAM_GAUSSIANMAP_OPTIMIZER_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <DataConverter.h>


#include <Gaussian/GaussianObject.h>

namespace ObjectSLAM {
	namespace GOMAP {
		namespace Optimizer {
			class ObjectOptimizer {
			public:
				static void ObjectPosOptimization(GOMAP::GaussianObject* pG);
			private:

			};
		}
	}
}


#endif
