#include <AssoFramePairData.h>

#include <KeyFrame.h>
#include <BoxFrame.h>

namespace ObjectSLAM {
	AssoFramePairData::AssoFramePairData(BoxFrame* pFrom, BoxFrame* pTo)
		:mpFrom(pFrom), mpTo(pTo), fromid(pFrom->mnId), toid(pTo->mnId), mpRaftIns(nullptr), mpSamIns(nullptr), mpSamIns2(nullptr)
		, mpPrevMapIns(nullptr), mpLocalMapIns(nullptr), mpFrameMapIns(nullptr)
		, mRaftAssoData(nullptr), mSamAssoData(nullptr), mFrameMapAssoData(nullptr), mLocalMapAssoData(nullptr)
	{

	}
	void AssoFramePairData::SetFromFrame(BoxFrame* pF) {
		mpFrom = pF;
		fromid = pF->mnId;
	}
	void AssoFramePairData::SetToFrame(BoxFrame* pF) {
		mpTo = pF;
		toid = pF->mnId;
	}
}
