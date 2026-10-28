#include <cmath>
int gcd(int u, int v) {
    if (u == 0) return std::abs(v);
    if (v == 0) return std::abs(u);
    u = std::abs(u); v = std::abs(v);
    int shift = 0;
    while (((u | v) & 1) == 0) { u >>= 1; v >>= 1; shift++; }
    while ((u & 1) == 0) u >>= 1;
    do {
        while ((v & 1) == 0) v >>= 1;
        if (u > v) { int temp = u; u = v; v = temp; }
        v -= u;
    } while (v != 0);
    return u << shift;
}
