#include <bits/stdc++.h>
using namespace std;
int main() {
    int n;
    cin >> n;
    long long total = 0, x;
    for (int i = 0; i < n; i++) {
        cin >> x;
        total += x;
    }
    cout << total << endl;
    return 0;
}
