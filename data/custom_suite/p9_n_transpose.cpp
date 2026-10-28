#include <vector>
using namespace std;
vector<vector<int>> transpose(const vector<vector<int>>& A) {
    int rows = A.size(), cols = A[0].size();
    vector<vector<int>> T(cols, vector<int>(rows, 0));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) T[j][i] = A[i][j];
    }
    return T;
}
