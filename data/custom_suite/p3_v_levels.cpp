#include <queue>
#include <vector>
using namespace std;
bool hasPath(const vector<vector<int>>& graph, int start, int target) {
    int n = graph.size();
    vector<int> dist(n, -1);
    queue<int> q;
    dist[start] = 0;
    q.push(start);
    while (!q.empty()) {
        int curr = q.front(); q.pop();
        if (curr == target) return true;
        for (int neighbor = 0; neighbor < n; neighbor++) {
            if (graph[curr][neighbor] == 1 && dist[neighbor] == -1) {
                dist[neighbor] = dist[curr] + 1;
                q.push(neighbor);
            }
        }
    }
    return dist[target] != -1;
}
