#include <bits/stdc++.h>
using namespace std;
int main() {
    int n;
    cin >> n;
    int mx = INT_MIN, x;
    for (int i = 0; i < n; i++) {
        cin >> x;
        if (x > mx) mx = x;
    }
    cout << mx << endl;
    return 0;
}
