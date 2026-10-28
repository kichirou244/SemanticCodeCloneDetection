#include <vector>
using namespace std;
vector<vector<int>> multiply(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int r1 = A.size(), c1 = A[0].size(), c2 = B[0].size();
    vector<vector<int>> C(r1, vector<int>(c2, 0));
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            for (int k = 0; k < c1; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}
