public class PrimeCheckLong {
    public static boolean isPrime(long n) {
        if (n <= 1L) return false;
        if (n <= 3L) return true;
        if (n % 2L == 0 || n % 3L == 0) return false;
        for (long i = 5L; i * i <= n; i += 6L) {
            if (n % i == 0 || n % (i + 2L) == 0) return false;
        }
        return true;
    }
}
