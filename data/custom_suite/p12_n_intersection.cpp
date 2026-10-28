#include <vector>
using namespace std;
vector<int> intersect(const vector<int>& a, const vector<int>& b) {
    vector<int> common;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) { common.push_back(a[i]); i++; j++; }
        else if (a[i] < b[j]) i++;
        else j++;
    }
    return common;
}
