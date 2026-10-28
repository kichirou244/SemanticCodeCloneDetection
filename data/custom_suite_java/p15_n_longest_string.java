public class LongestStringFinder {
    public static String longestCommonPrefix(String[] strs) { // tim chuoi dai nhat
        if (strs == null || strs.length == 0) return "";
        String longest = strs[0];
        for (int i = 1; i < strs.length; i++) {
            if (strs[i].length() > longest.length()) longest = strs[i];
        }
        return longest;
    }
}
