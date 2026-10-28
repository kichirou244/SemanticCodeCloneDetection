#include <vector>
using namespace std;
vector<vector<int>> multiply(const vector<vector<int>>& A, const vector<vector<int>>& B) {
    int r1 = A.size(), c1 = A[0].size(), c2 = B[0].size();
    vector<vector<int>> C(r1, vector<int>(c2, 0));
    int i = 0;
    while (i < r1) {
        int j = 0;
        while (j < c2) {
            int k = 0;
            while (k < c1) { C[i][j] += A[i][k] * B[k][j]; k++; }
            j++;
        }
        i++;
    }
    return C;
}
