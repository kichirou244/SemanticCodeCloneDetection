#include <bits/stdc++.h>
using namespace std;
int main() {
    int n;
    cin >> n;
    long long prod = 1;
    for (int i = 1; i <= n; i++) {
        prod *= i;
    }
    cout << prod << endl;
    return 0;
}
