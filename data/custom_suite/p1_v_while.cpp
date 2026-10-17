#include <bits/stdc++.h>
using namespace std;
int main() {
    int N, total = 0;
    cin >> N;
    int k = 1;
    while (k <= N) {
        total = total + k;
        k++;
    }
    cout << total << endl;
    return 0;
}
