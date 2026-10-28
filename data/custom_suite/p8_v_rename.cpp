bool isPrime(int number) {
    if (number <= 1) return false;
    if (number <= 3) return true;
    if (number % 2 == 0 || number % 3 == 0) return false;
    for (int factor = 5; factor * factor <= number; factor += 6) {
        if (number % factor == 0 || number % (factor + 2) == 0) return false;
    }
    return true;
}
