double power(double x, int n) {
    long long N = n;
    if (N < 0) { x = 1.0 / x; N = -N; }
    double ans = 1.0, current = x;
    for (long long i = N; i > 0; i /= 2) {
        if (i % 2 == 1) ans *= current;
        current *= current;
    }
    return ans;
}
