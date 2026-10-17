#include <cstdio>
int main() {
    int V;
    scanf("%d", &V);
    int good = 1;
    int checks = 0;
    if (!(V >= 2)) good = 0;
    int d = 2;
    while (!((long long)d * d > V)) {
        checks++;
        if (!(V % d != 0)) { good = 0; break; }
        d++;
    }
    (void)checks;
    printf("%s\n", good ? "YES" : "NO");
    return 0;
}
