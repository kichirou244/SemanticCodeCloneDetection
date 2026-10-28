import java.util.Arrays;

public class DynamicArrayList {
    private int[] data;
    private int size;

    public DynamicArrayList() {
        data = new int[8];
        size = 0;
    }

    public void add(int val) {
        if (size == data.length) {
            data = Arrays.copyOf(data, data.length * 2);
        }
        data[size++] = val;
    }

    public boolean contains(int target) {
        for (int i = 0; i < size; i++) {
            if (data[i] == target) return true;
        }
        return false;
    }

    public int sum() {
        int total = 0;
        for (int i = 0; i < size; i++) {
            total += data[i];
        }
        return total;
    }
}
