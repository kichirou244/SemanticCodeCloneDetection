public class LcpCaseSensitive {
    public static String longestCommonPrefix(String[] strs) {
        if (strs.length == 0) return "";
        StringBuilder sb = new StringBuilder(strs[0]);
        for (int i = 1; i < strs.length; i++) {
            while (!strs[i].startsWith(sb.toString())) {
                sb.deleteCharAt(sb.length() - 1);
                if (sb.length() == 0) return "";
            }
        }
        return sb.toString();
    }
}
