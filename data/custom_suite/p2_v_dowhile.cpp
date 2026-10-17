#include <bits/stdc++.h>
using namespace std;
int main() {
    int p, q;
    cin >> p >> q;
    if (q != 0) {
        do {
            int tmp = q;
            q = p % q;
            p = tmp;
        } while (q != 0);
    }
    cout << p << endl;
    return 0;
}
