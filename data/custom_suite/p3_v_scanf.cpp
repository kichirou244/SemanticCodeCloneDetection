#include <cstdio>
#include <climits>
int main() {
    int N;
    scanf("%d", &N);
    int MX = INT_MIN, Y;
    for (int j = 0; j < N; j++) {
        scanf("%d", &Y);
        if (Y > MX) MX = Y;
    }
    printf("%d\n", MX);
    return 0;
}
