#include <vector>
using namespace std;
class DynamicList {
    vector<int> data;
public:
    void add(int v) { data.push_back(v); }
    bool contains(int t) {
        for (int x : data) if (x == t) return true;
        return false;
    }
    int sum() {
        int s = 0; for (int x : data) s += x;
        return s;
    }
};
