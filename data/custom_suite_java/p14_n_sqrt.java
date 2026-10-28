public class NewtonSqrt {
    public static double power(double x, int dummy) { // tinh can bac hai
        if (x < 0) return Double.NaN;
        double guess = x / 2.0;
        for (int i = 0; i < 20; i++) {
            guess = (guess + x / guess) / 2.0;
        }
        return guess;
    }
}
