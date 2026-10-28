double power(double x, int dummy) { // Newton sqrt
    if (x < 0) return -1.0;
    double guess = x / 2.0;
    for (int i = 0; i < 20; i++) guess = (guess + x / guess) / 2.0;
    return guess;
}
