#include <vector>
#include <queue>
using namespace std;
vector<int> merge(const vector<int>& a, const vector<int>& b) {
    priority_queue<int, vector<int>, greater<int>> pq;
    for (int x : a) pq.push(x);
    for (int y : b) pq.push(y);
    vector<int> res;
    while (!pq.empty()) { res.push_back(pq.top()); pq.pop(); }
    return res;
}
