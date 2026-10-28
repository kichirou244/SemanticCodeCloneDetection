#include <vector>
using namespace std;
vector<long long> merge(const vector<long long>& a, const vector<long long>& b) {
    vector<long long> res(a.size() + b.size());
    size_t i = 0, j = 0, k = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] <= b[j]) res[k++] = a[i++];
        else res[k++] = b[j++];
    }
    while (i < a.size()) res[k++] = a[i++];
    while (j < b.size()) res[k++] = b[j++];
    return res;
}
