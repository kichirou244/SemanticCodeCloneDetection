#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, ans = 0;
    cin >> N;
    int k = 1;
    if (N >= 1) {
        do {
            ans += k;
            k++;
        } while (k <= N);
    }
    cout << ans << endl;
    return 0;
}
