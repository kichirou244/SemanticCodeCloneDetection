import java.util.Arrays;

public class StringAnagramChecker {
    // Kiem tra 2 chuoi co cung tap ky tu khong
    public static boolean isAnagram(String firstStr, String secondStr) {
        if (firstStr.length() != secondStr.length()) return false;
        char[] chars1 = firstStr.toCharArray();
        char[] chars2 = secondStr.toCharArray();
        Arrays.sort(chars1);
        Arrays.sort(chars2);
        return Arrays.equals(chars1, chars2);
    }
}
