double getFib(int n) {
    if (n <= 1) return (double)n;
    double a = 0.0, b = 1.0;
    for (int i = 2; i <= n; i++) {
        double temp = a + b; a = b; b = temp;
    }
    return b;
}
