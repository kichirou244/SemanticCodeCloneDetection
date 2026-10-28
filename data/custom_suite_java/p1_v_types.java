public class BubbleSorterGeneric {
    public static void sort(Integer[] elements) {
        int count = elements.length;
        for (int i = 0; i < count - 1; i++) {
            for (int j = 0; j < count - i - 1; j++) {
                if (elements[j].compareTo(elements[j + 1]) > 0) {
                    Integer temporary = elements[j];
                    elements[j] = elements[j + 1];
                    elements[j + 1] = temporary;
                }
            }
        }
    }
}
