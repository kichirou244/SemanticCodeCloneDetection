double power(double x, int n) {
    if (n == 0) return 1.0;
    long long N = n;
    if (N < 0) { x = 1.0 / x; N = -N; }
    double ans = 1.0;
    while (N > 0) { ans *= x; N--; }
    return ans;
}
