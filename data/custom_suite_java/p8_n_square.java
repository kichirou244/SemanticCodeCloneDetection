public class PerfectSquareCheck {
    public static boolean isPerfectSquare(int n) {
        if (n < 0) return false;
        int root = (int) Math.sqrt(n);
        return root * root == n;
    }
}
