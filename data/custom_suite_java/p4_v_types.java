public class CharArrayPalindrome {
    public static boolean isPalindrome(String text) {
        char[] characters = text.toCharArray();
        int i = 0;
        int j = characters.length - 1;
        while (i < j) {
            if (characters[i] != characters[j]) {
                return false;
            }
            i++;
            j--;
        }
        return true;
    }
}
