#include <bits/stdc++.h>
using namespace std;
int main() {
    int N;
    cin >> N;
    int cur = INT_MIN, val;
    int k = 0;
    if (k < N) {
        do {
            cin >> val;
            if (val > cur) cur = val;
            k++;
        } while (k < N);
    }
    cout << cur << endl;
    return 0;
}
