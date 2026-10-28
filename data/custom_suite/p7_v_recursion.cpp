#include <vector>
using namespace std;
long long fibMemo(int n, vector<long long>& memo) {
    if (n <= 1) return n;
    if (memo[n] != 0) return memo[n];
    return memo[n] = fibMemo(n - 1, memo) + fibMemo(n - 2, memo);
}
long long getFib(int n) {
    vector<long long> memo(n + 2, 0);
    return fibMemo(n, memo);
}
