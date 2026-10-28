#include <vector>
using namespace std;
vector<int> multiply(const vector<int>& A, const vector<int>& B, int r1, int c1, int c2) {
    vector<int> C(r1 * c2, 0);
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            int sum = 0;
            for (int k = 0; k < c1; k++) sum += A[i * c1 + k] * B[k * c2 + j];
            C[i * c2 + j] = sum;
        }
    }
    return C;
}
