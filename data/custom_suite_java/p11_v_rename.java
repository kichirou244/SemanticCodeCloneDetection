public class KadaneAlgorithm {
    // Tim tong day con lien tiep lon nhat
    public static int maxSubArray(int[] numbers) {
        int bestSum = numbers[0];
        int runningSum = numbers[0];
        for (int idx = 1; idx < numbers.length; idx++) {
            runningSum = Math.max(numbers[idx], runningSum + numbers[idx]);
            bestSum = Math.max(bestSum, runningSum);
        }
        return bestSum;
    }
}
