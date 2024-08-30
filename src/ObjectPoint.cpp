#include <ObjectPoint.h>

namespace ObjectSLAM {
	std::atomic<int> ObjectPoint::nObjPointId = 0;

	ObjectPoint::ObjectPoint() : BaseSLAM::AbstractData(), BoxObjObservation() {
		mnId = (++nObjPointId);
	}
	ObjectPoint::ObjectPoint(BoundingBox* _ref) : BaseSLAM::AbstractData((BaseSLAM::AbstractFrame*)_ref), BoxObjObservation(_ref) {
		mnId = (++nObjPointId);
	}
	ObjectPoint::ObjectPoint(SegInstance* _ref) : BaseSLAM::AbstractData((BaseSLAM::AbstractFrame*)_ref), BoxObjObservation() {
		mnId = (++nObjPointId);
	}
}