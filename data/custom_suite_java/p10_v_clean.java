import java.util.Arrays;

public class AnagramCleaned {
    public static boolean isAnagram(String s1, String s2) {
        String c1 = s1.replaceAll("\\s+", "");
        String c2 = s2.replaceAll("\\s+", "");
        if (c1.length() != c2.length()) return false;
        char[] a = c1.toCharArray();
        char[] b = c2.toCharArray();
        Arrays.sort(a);
        Arrays.sort(b);
        return Arrays.equals(a, b);
    }
}
