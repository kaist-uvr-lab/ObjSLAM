#include <AssoFramePairData.h>

#include <KeyFrame.h>
#include <BoxFrame.h>

namespace ObjectSLAM {
	AssoFramePairData::AssoFramePairData(BoxFrame* pFrom, BoxFrame* pTo)
		:mpFrom(pFrom), mpTo(pTo), fromid(pFrom->mnId), toid(pTo->mnId), mpRaftIns(nullptr), mpSamIns(nullptr)
		, mpPrevMapIns(nullptr), mpLocalMapIns(nullptr)
	{

	}
}
