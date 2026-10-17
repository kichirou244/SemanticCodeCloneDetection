#include <bits/stdc++.h>
using namespace std;
int main() {
    int V;
    cin >> V;
    bool ok = true;
    if (V < 2) ok = false;
    int k = 2;
    if ((long long)k * k <= V) {
        do {
            if (V % k == 0) { ok = false; break; }
            k++;
        } while ((long long)k * k <= V);
    }
    cout << (ok ? "YES" : "NO") << endl;
    return 0;
}
