#include <queue>
#include <vector>
using namespace std;
bool hasPath(const vector<vector<int>>& adjMat, int src, int dest) {
    int total = adjMat.size();
    vector<bool> seen(total, false);
    queue<int> worklist;
    seen[src] = true;
    worklist.push(src);
    while (!worklist.empty()) {
        int node = worklist.front(); worklist.pop();
        if (node == dest) return true;
        for (int adj = 0; adj < total; adj++) {
            if (adjMat[node][adj] == 1 && !seen[adj]) {
                seen[adj] = true;
                worklist.push(adj);
            }
        }
    }
    return false;
}
