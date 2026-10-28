public class BinaryExponentiation {
    public static double power(double x, int n) {
        long N = n;
        if (N < 0) { x = 1 / x; N = -N; }
        double ans = 1.0;
        double current = x;
        for (long i = N; i > 0; i /= 2) {
            if (i % 2 == 1) ans *= current;
            current *= current;
        }
        return ans;
    }
}
