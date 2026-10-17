#include <bits/stdc++.h>
using namespace std;
int main() {
    int n;
    cin >> n;
    bool square = false;
    for (int i = 1; (long long)i * i <= n; i++) {
        if (i * i == n) { square = true; break; }
    }
    cout << (square ? "YES" : "NO") << endl;
    return 0;
}
