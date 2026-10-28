public class FibonacciMemo {
    public static long getFib(int n) {
        long[] memo = new long[n + 2];
        return fib(n, memo);
    }
    private static long fib(int n, long[] memo) {
        if (n <= 1) return n;
        if (memo[n] != 0) return memo[n];
        memo[n] = fib(n - 1, memo) + fib(n - 2, memo);
        return memo[n];
    }
}
