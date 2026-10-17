#include <cstdio>
#include <climits>
int main() {
    int N;
    scanf("%d", &N);
    int top = INT_MIN, z;
    int cnt = 0;
    int seen = 0;
    while (!(cnt >= N)) {
        scanf("%d", &z);
        if (!(z <= top)) top = z;
        cnt++;
        seen++;
    }
    (void)seen;
    printf("%d\n", top);
    return 0;
}
