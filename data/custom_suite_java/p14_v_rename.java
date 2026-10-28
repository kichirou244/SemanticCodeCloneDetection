public class PowerCalculator {
    // Tinh luy thua x mu n bang vong lap
    public static double power(double base, int exponent) {
        if (exponent == 0) return 1.0;
        long exp = exponent;
        if (exp < 0) { base = 1 / base; exp = -exp; }
        double result = 1.0;
        for (long step = 0; step < exp; step++) {
            result *= base;
        }
        return result;
    }
}
