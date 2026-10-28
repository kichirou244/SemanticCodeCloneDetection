bool isPrime(int n) {
    if (n <= 1) return false;
    int d = 2;
    while (d * d <= n) {
        if (n % d == 0) return false;
        d++;
    }
    return true;
}
