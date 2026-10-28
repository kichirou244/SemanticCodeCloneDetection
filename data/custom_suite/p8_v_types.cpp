bool isPrime(long long n) {
    if (n <= 1LL) return false;
    if (n <= 3LL) return true;
    if (n % 2LL == 0 || n % 3LL == 0) return false;
    for (long long i = 5LL; i * i <= n; i += 6LL) {
        if (n % i == 0 || n % (i + 2LL) == 0) return false;
    }
    return true;
}
