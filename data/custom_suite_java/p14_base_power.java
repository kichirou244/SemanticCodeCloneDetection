public class PowerFunction {
    public static double power(double x, int n) {
        if (n == 0) return 1.0;
        long N = n;
        if (N < 0) { x = 1 / x; N = -N; }
        double ans = 1.0;
        for (long i = 0; i < N; i++) {
            ans *= x;
        }
        return ans;
    }
}
