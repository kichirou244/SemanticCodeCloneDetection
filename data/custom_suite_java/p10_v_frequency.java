public class AnagramFrequency {
    public static boolean isAnagram(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        int[] counts = new int[256];
        for (int i = 0; i < s1.length(); i++) {
            counts[s1.charAt(i)]++;
            counts[s2.charAt(i)]--;
        }
        for (int c : counts) {
            if (c != 0) return false;
        }
        return true;
    }
}
