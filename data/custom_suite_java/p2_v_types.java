public class GcdLong {
    public static long gcd(long x, long y) {
        while (y != 0L) {
            long r = x % y;
            x = y;
            y = r;
        }
        return Math.abs(x);
    }
}
