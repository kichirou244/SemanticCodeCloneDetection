public class MinSubarray {
    public static int minSubArray(int[] nums) {
        int minSoFar = nums[0];
        int currMin = nums[0];
        for (int i = 1; i < nums.length; i++) {
            currMin = Math.min(nums[i], currMin + nums[i]);
            minSoFar = Math.min(minSoFar, currMin);
        }
        return minSoFar;
    }
}
