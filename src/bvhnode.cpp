#include "bvhnode.hpp"

void BuildNode::computeBB(std::vector<TriRef> &refs) {
	assert(iStart <= iEnd); // range must be non-empty
	for (U32 i = iStart; i <= iEnd; i++) {
		box.expand(refs[i].box);
	}
}

void SBVHNode::deleteTree() {
	if (!this->isLeaf()) {
		leftChild->deleteTree();
		rightChild->deleteTree();
	}

	delete this;
}