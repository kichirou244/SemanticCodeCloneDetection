#include <bits/stdc++.h>
using namespace std;
int main() {
    int N;
    cin >> N;
    bool isPrime = true;
    if (N < 2) isPrime = false;
    int d = 2;
    while ((long long)d * d <= N) {
        if (N % d == 0) { isPrime = false; break; }
        d++;
    }
    cout << (isPrime ? "YES" : "NO") << endl;
    return 0;
}
