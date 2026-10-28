import java.util.Arrays;

public class AnagramCaseInsensitive {
    public static boolean isAnagram(String s1, String s2) {
        String lower1 = s1.toLowerCase();
        String lower2 = s2.toLowerCase();
        if (lower1.length() != lower2.length()) return false;
        char[] a1 = lower1.toCharArray();
        char[] a2 = lower2.toCharArray();
        Arrays.sort(a1);
        Arrays.sort(a2);
        return Arrays.equals(a1, a2);
    }
}
