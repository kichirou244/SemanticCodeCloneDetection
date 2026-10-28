#include <vector>
using namespace std;
vector<vector<int>> multiply(const vector<vector<int>>& firstMat, const vector<vector<int>>& secondMat) {
    int rowsA = firstMat.size(), colsA = firstMat[0].size(), colsB = secondMat[0].size();
    vector<vector<int>> res(rowsA, vector<int>(colsB, 0));
    for (int r = 0; r < rowsA; r++) {
        for (int c = 0; c < colsB; c++) {
            for (int p = 0; p < colsA; p++) {
                res[r][c] += firstMat[r][p] * secondMat[p][c];
            }
        }
    }
    return res;
}
