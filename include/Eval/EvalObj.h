#ifndef EVALUATION_OBJECT_H
#define EVALUATION_OBJECT_H
#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#include <ConcurrentSet.h>
#include <ConcurrentMap.h>
#include <ConcurrentVector.h>

namespace ObjectSLAM {

	namespace Evaluation {
		class EvalObj {
		public:
			EvalObj(){

			}

			virtual ~EvalObj() {}
		public:
			//에러 기록

		private:

		};
	}
}

#endif