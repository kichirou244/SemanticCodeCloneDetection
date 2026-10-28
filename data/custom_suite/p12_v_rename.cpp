#include <vector>
using namespace std;
vector<int> merge(const vector<int>& leftArr, const vector<int>& rightArr) {
    vector<int> merged(leftArr.size() + rightArr.size());
    size_t p1 = 0, p2 = 0, idx = 0;
    while (p1 < leftArr.size() && p2 < rightArr.size()) {
        if (leftArr[p1] <= rightArr[p2]) merged[idx++] = leftArr[p1++];
        else merged[idx++] = rightArr[p2++];
    }
    while (p1 < leftArr.size()) merged[idx++] = leftArr[p1++];
    while (p2 < rightArr.size()) merged[idx++] = rightArr[p2++];
    return merged;
}
