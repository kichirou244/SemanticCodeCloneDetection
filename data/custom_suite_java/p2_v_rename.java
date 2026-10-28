public class EuclidGcd {
    // Tim uoc chung lon nhat cua 2 so nguyen bang giai thuat Euclid
    public static int gcd(int firstNumber, int secondNumber) {
        while (secondNumber != 0) {
            int rem = firstNumber % secondNumber;
            firstNumber = secondNumber;
            secondNumber = rem;
        }
        return Math.abs(firstNumber);
    }
}
