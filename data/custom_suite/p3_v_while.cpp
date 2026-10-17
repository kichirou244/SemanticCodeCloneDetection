#include <bits/stdc++.h>
using namespace std;
int main() {
    int N;
    cin >> N;
    int best = INT_MIN, v;
    int k = 0;
    while (k < N) {
        cin >> v;
        if (v > best) best = v;
        k++;
    }
    cout << best << endl;
    return 0;
}
