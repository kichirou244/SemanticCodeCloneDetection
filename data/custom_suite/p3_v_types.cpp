#include <deque>
#include <vector>
using namespace std;
bool hasPath(const vector<vector<int>>& graph, int start, int target) {
    int n = graph.size();
    vector<char> visited(n, 0);
    deque<int> q;
    visited[start] = 1;
    q.push_back(start);
    while (!q.empty()) {
        int curr = q.front(); q.pop_front();
        if (curr == target) return true;
        for (int neighbor = 0; neighbor < n; neighbor++) {
            if (graph[curr][neighbor] == 1 && !visited[neighbor]) {
                visited[neighbor] = 1;
                q.push_back(neighbor);
            }
        }
    }
    return false;
}
