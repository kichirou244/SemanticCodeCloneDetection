#include <cstdio>
int main() {
    int A, B;
    scanf("%d %d", &A, &B);
    while (B != 0) {
        int T = B;
        B = A % B;
        A = T;
    }
    printf("%d\n", A);
    return 0;
}
