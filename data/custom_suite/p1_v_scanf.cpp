#include <cstdio>
int main() {
    int N, S = 0;
    scanf("%d", &N);
    for (int j = 1; j <= N; j++) {
        S += j;
    }
    printf("%d\n", S);
    return 0;
}
