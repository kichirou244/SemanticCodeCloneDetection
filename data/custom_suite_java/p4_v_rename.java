public class StringPalindrome {
    // Kiem tra chuoi doi xung bang hai con tro chay tu hai dau
    public static boolean isPalindrome(String inputStr) {
        int startIdx = 0;
        int endIdx = inputStr.length() - 1;
        while (startIdx < endIdx) {
            if (inputStr.charAt(startIdx) != inputStr.charAt(endIdx)) {
                return false;
            }
            startIdx++;
            endIdx--;
        }
        return true;
    }
}
