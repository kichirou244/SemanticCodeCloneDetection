public class GreatestCommonDivisor {
    public static int gcd(int a, int b) {
        while (b != 0) {
            int remainder = a % b;
            a = b;
            b = remainder;
        }
        return Math.abs(a);
    }
}
