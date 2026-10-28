public class PrimeTrialDivision {
    public static boolean isPrime(int n) {
        if (n < 2) return false;
        int limit = (int) Math.sqrt(n);
        for (int i = 2; i <= limit; i++) {
            if (n % i == 0) return false;
        }
        return true;
    }
}
