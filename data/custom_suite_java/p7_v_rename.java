public class FibCalculator {
    // Tinh so Fibonacci thu n bang vong lap
    public static long getFib(int num) {
        if (num <= 1) return num;
        long first = 0, second = 1;
        for (int step = 2; step <= num; step++) {
            long nextVal = first + second;
            first = second;
            second = nextVal;
        }
        return second;
    }
}
