long long power(long long base, int exp) {
    long long res = 1LL;
    for (int i = 0; i < exp; i++) res *= base;
    return res;
}
