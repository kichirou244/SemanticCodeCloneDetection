public class IntPower {
    public static long power(long base, int exp) {
        long res = 1L;
        for (int i = 0; i < exp; i++) {
            res *= base;
        }
        return res;
    }
}
