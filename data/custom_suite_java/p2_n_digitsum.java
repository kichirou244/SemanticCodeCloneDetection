public class DigitSum {
    public static int sumDigits(int number) {
        int sum = 0;
        int n = Math.abs(number);
        while (n != 0) {
            int remainder = n % 10;
            sum += remainder;
            n = n / 10;
        }
        return sum;
    }
}
