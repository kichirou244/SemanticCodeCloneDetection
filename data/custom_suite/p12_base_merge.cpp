#include <vector>
using namespace std;
vector<int> merge(const vector<int>& a, const vector<int>& b) {
    vector<int> res(a.size() + b.size());
    size_t i = 0, j = 0, k = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] <= b[j]) res[k++] = a[i++];
        else res[k++] = b[j++];
    }
    while (i < a.size()) res[k++] = a[i++];
    while (j < b.size()) res[k++] = b[j++];
    return res;
}
