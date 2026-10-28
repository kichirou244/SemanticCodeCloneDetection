public class MaxSubarrayWhile {
    public static int maxSubArray(int[] nums) {
        int maxSoFar = nums[0];
        int currMax = nums[0];
        int i = 1;
        while (i < nums.length) {
            if (currMax < 0) currMax = nums[i];
            else currMax += nums[i];
            if (currMax > maxSoFar) maxSoFar = currMax;
            i++;
        }
        return maxSoFar;
    }
}
