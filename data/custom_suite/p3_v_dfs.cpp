#include <vector>
using namespace std;
bool dfs(const vector<vector<int>>& g, int curr, int target, vector<bool>& visited) {
    if (curr == target) return true;
    visited[curr] = true;
    for (int next = 0; next < (int)g.size(); next++) {
        if (g[curr][next] == 1 && !visited[next]) {
            if (dfs(g, next, target, visited)) return true;
        }
    }
    return false;
}
bool hasPath(const vector<vector<int>>& graph, int start, int target) {
    vector<bool> visited(graph.size(), false);
    return dfs(graph, start, target, visited);
}
