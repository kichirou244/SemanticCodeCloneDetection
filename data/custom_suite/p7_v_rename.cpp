long long getFib(int num) {
    if (num <= 1) return num;
    long long first = 0, second = 1;
    for (int step = 2; step <= num; step++) {
        long long nextVal = first + second; first = second; second = nextVal;
    }
    return second;
}
