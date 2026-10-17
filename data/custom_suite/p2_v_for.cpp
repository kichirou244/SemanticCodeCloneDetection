#include <bits/stdc++.h>
using namespace std;
int main() {
    int x, y;
    cin >> x >> y;
    for (; y != 0;) {
        int tmp = y;
        y = x % y;
        x = tmp;
    }
    cout << x << endl;
    return 0;
}
