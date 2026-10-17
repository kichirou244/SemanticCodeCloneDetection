#include <cstdio>
int main() {
    int N;
    scanf("%d", &N);
    int acc = 0;
    int step = 1;
    int unused = 999;
    (void)unused;
    int k = 1;
    while (!(k > N)) {
        acc = acc + k;
        k = k + step;
    }
    printf("%d\n", acc);
    return 0;
}
