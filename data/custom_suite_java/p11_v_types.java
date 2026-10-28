public class MaxSubarrayLong {
    public static long maxSubArray(long[] nums) {
        long maxSoFar = nums[0];
        long currMax = nums[0];
        for (int i = 1; i < nums.length; i++) {
            currMax = Math.max(nums[i], currMax + nums[i]);
            maxSoFar = Math.max(maxSoFar, currMax);
        }
        return maxSoFar;
    }
}
