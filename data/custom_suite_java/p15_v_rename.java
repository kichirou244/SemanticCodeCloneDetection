public class LcpSolver {
    // Tim tien to chung dai nhat cua mang chuoi
    public static String longestCommonPrefix(String[] words) {
        if (words == null || words.length == 0) return "";
        String commonPref = words[0];
        for (int idx = 1; idx < words.length; idx++) {
            while (words[idx].indexOf(commonPref) != 0) {
                commonPref = commonPref.substring(0, commonPref.length() - 1);
                if (commonPref.isEmpty()) return "";
            }
        }
        return commonPref;
    }
}
