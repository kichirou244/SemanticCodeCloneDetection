#include <cstdio>
int main() {
    int N;
    scanf("%d", &N);
    int prime = 1;
    if (N < 2) prime = 0;
    for (int j = 2; (long long)j * j <= N; j++) {
        if (N % j == 0) { prime = 0; break; }
    }
    printf("%s\n", prime ? "YES" : "NO");
    return 0;
}
