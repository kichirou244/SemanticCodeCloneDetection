double power(double base, int exponent) {
    if (exponent == 0) return 1.0;
    long long exp = exponent;
    if (exp < 0) { base = 1.0 / base; exp = -exp; }
    double result = 1.0;
    for (long long step = 0; step < exp; step++) result *= base;
    return result;
}
