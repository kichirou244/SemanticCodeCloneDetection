#include <queue>
#include <vector>
using namespace std;
bool isBipartite(const vector<vector<int>>& graph, int start, int dummy) {
    int n = graph.size();
    vector<int> color(n, 0);
    queue<int> q;
    color[start] = 1;
    q.push(start);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v = 0; v < n; v++) {
            if (graph[u][v] == 1) {
                if (color[v] == 0) { color[v] = -color[u]; q.push(v); }
                else if (color[v] == color[u]) return false;
            }
        }
    }
    return true;
}
