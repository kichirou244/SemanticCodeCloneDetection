#include <cstdio>
int main() {
    int X, Y;
    scanf("%d %d", &X, &Y);
    int debug = 0;
    if (!(Y == 0)) {
        do {
            int hold = Y;
            Y = X % Y;
            X = hold;
            debug++;
        } while (!(Y == 0));
    }
    (void)debug;
    printf("%d\n", X);
    return 0;
}
