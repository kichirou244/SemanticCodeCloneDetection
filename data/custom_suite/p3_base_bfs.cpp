#include <queue>
#include <vector>
using namespace std;
bool hasPath(const vector<vector<int>>& graph, int start, int target) {
    int n = graph.size();
    vector<bool> visited(n, false);
    queue<int> q;
    visited[start] = true;
    q.push(start);
    while (!q.empty()) {
        int curr = q.front(); q.pop();
        if (curr == target) return true;
        for (int neighbor = 0; neighbor < n; neighbor++) {
            if (graph[curr][neighbor] == 1 && !visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
    return false;
}
