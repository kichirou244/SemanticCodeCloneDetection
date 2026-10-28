#include <vector>
using namespace std;
vector<int> merge(const vector<int>& a, const vector<int>& b) {
    size_t n = a.size(), m = b.size();
    vector<int> res(n + m);
    size_t i = 0, j = 0;
    for (size_t k = 0; k < n + m; k++) {
        if (i >= n) res[k] = b[j++];
        else if (j >= m) res[k] = a[i++];
        else if (a[i] <= b[j]) res[k] = a[i++];
        else res[k] = b[j++];
    }
    return res;
}
