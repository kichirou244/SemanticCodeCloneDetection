"""Sinh day du 15 bai toan (90 file Java) va manifest pairs.jsonl cho data/custom_suite_java/
Ti le chuan 1:5 (120 Clones : 600 Non-clones = 720 cap danh gia).
"""

from __future__ import annotations
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "custom_suite_java"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# P6: Binary Search
P6_FILES = {
    "p6_base_bsearch.java": """public class BinarySearch {
    public static int search(int[] arr, int target) {
        int low = 0;
        int high = arr.length - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (arr[mid] == target) return mid;
            if (arr[mid] < target) low = mid + 1;
            else high = mid - 1;
        }
        return -1;
    }
}""",
    "p6_v_rename.java": """public class BSearchAlgorithm {
    // Tim kiem nhi phan tren day da sap xep
    public static int search(int[] numbers, int key) {
        int leftIdx = 0;
        int rightIdx = numbers.length - 1;
        while (leftIdx <= rightIdx) {
            int middle = leftIdx + (rightIdx - leftIdx) / 2;
            if (numbers[middle] == key) return middle;
            if (numbers[middle] < key) leftIdx = middle + 1;
            else rightIdx = middle - 1;
        }
        return -1;
    }
}""",
    "p6_v_types.java": """public class BinarySearchLong {
    public static int search(long[] arr, long target) {
        int low = 0;
        int high = arr.length - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (arr[mid] == target) return mid;
            if (arr[mid] < target) low = mid + 1;
            else high = mid - 1;
        }
        return -1;
    }
}""",
    "p6_v_recursion.java": """public class RecursiveBinarySearch {
    public static int search(int[] arr, int target) {
        return bsearch(arr, target, 0, arr.length - 1);
    }
    private static int bsearch(int[] arr, int target, int low, int high) {
        if (low > high) return -1;
        int mid = low + (high - low) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) return bsearch(arr, target, mid + 1, high);
        return bsearch(arr, target, low, mid - 1);
    }
}""",
    "p6_v_exponential.java": """public class ExponentialSearch {
    public static int search(int[] arr, int target) {
        int n = arr.length;
        if (n == 0) return -1;
        if (arr[0] == target) return 0;
        int i = 1;
        while (i < n && arr[i] <= target) i = i * 2;
        int low = i / 2;
        int high = Math.min(i, n - 1);
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (arr[mid] == target) return mid;
            if (arr[mid] < target) low = mid + 1;
            else high = mid - 1;
        }
        return -1;
    }
}""",
    "p6_n_linear_count.java": """public class LinearSearchCounter {
    public static int countOccurrences(int[] arr, int target) {
        int count = 0;
        for (int i = 0; i < arr.length; i++) {
            if (arr[i] == target) count++;
        }
        return count;
    }
}""",
}

# P7: Fibonacci Number
P7_FILES = {
    "p7_base_fib.java": """public class Fibonacci {
    public static long getFib(int n) {
        if (n <= 1) return n;
        long a = 0, b = 1;
        for (int i = 2; i <= n; i++) {
            long temp = a + b;
            a = b;
            b = temp;
        }
        return b;
    }
}""",
    "p7_v_rename.java": """public class FibCalculator {
    // Tinh so Fibonacci thu n bang vong lap
    public static long getFib(int num) {
        if (num <= 1) return num;
        long first = 0, second = 1;
        for (int step = 2; step <= num; step++) {
            long nextVal = first + second;
            first = second;
            second = nextVal;
        }
        return second;
    }
}""",
    "p7_v_types.java": """import java.math.BigInteger;

public class FibonacciBigInt {
    public static BigInteger getFib(int n) {
        if (n <= 1) return BigInteger.valueOf(n);
        BigInteger a = BigInteger.ZERO, b = BigInteger.ONE;
        for (int i = 2; i <= n; i++) {
            BigInteger temp = a.add(b);
            a = b;
            b = temp;
        }
        return b;
    }
}""",
    "p7_v_recursion.java": """public class FibonacciMemo {
    public static long getFib(int n) {
        long[] memo = new long[n + 2];
        return fib(n, memo);
    }
    private static long fib(int n, long[] memo) {
        if (n <= 1) return n;
        if (memo[n] != 0) return memo[n];
        memo[n] = fib(n - 1, memo) + fib(n - 2, memo);
        return memo[n];
    }
}""",
    "p7_v_matrix.java": """public class FibonacciMatrix {
    public static long getFib(int n) {
        if (n <= 1) return n;
        long[][] F = {{1, 1}, {1, 0}};
        power(F, n - 1);
        return F[0][0];
    }
    private static void multiply(long[][] F, long[][] M) {
        long x = F[0][0]*M[0][0] + F[0][1]*M[1][0];
        long y = F[0][0]*M[0][1] + F[0][1]*M[1][1];
        long z = F[1][0]*M[0][0] + F[1][1]*M[1][0];
        long w = F[1][0]*M[0][1] + F[1][1]*M[1][1];
        F[0][0] = x; F[0][1] = y; F[1][0] = z; F[1][1] = w;
    }
    private static void power(long[][] F, int n) {
        if (n == 0 || n == 1) return;
        long[][] M = {{1, 1}, {1, 0}};
        power(F, n / 2);
        multiply(F, F);
        if (n % 2 != 0) multiply(F, M);
    }
}""",
    "p7_n_factorial.java": """public class Factorial {
    public static long getFactorial(int n) {
        if (n <= 1) return 1;
        long result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }
}""",
}

# P8: Prime Number Check
P8_FILES = {
    "p8_base_prime.java": """public class PrimeCheck {
    public static boolean isPrime(int n) {
        if (n <= 1) return false;
        if (n <= 3) return true;
        if (n % 2 == 0 || n % 3 == 0) return false;
        for (int i = 5; i * i <= n; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0) return false;
        }
        return true;
    }
}""",
    "p8_v_rename.java": """public class PrimeValidator {
    // Kiem tra so nguyen to toi uu 6k +- 1
    public static boolean isPrime(int number) {
        if (number <= 1) return false;
        if (number <= 3) return true;
        if (number % 2 == 0 || number % 3 == 0) return false;
        for (int factor = 5; factor * factor <= number; factor += 6) {
            if (number % factor == 0 || number % (factor + 2) == 0) return false;
        }
        return true;
    }
}""",
    "p8_v_types.java": """public class PrimeCheckLong {
    public static boolean isPrime(long n) {
        if (n <= 1L) return false;
        if (n <= 3L) return true;
        if (n % 2L == 0 || n % 3L == 0) return false;
        for (long i = 5L; i * i <= n; i += 6L) {
            if (n % i == 0 || n % (i + 2L) == 0) return false;
        }
        return true;
    }
}""",
    "p8_v_while.java": """public class PrimeCheckWhile {
    public static boolean isPrime(int n) {
        if (n <= 1) return false;
        int d = 2;
        while (d * d <= n) {
            if (n % d == 0) return false;
            d++;
        }
        return true;
    }
}""",
    "p8_v_sieve.java": """public class PrimeTrialDivision {
    public static boolean isPrime(int n) {
        if (n < 2) return false;
        int limit = (int) Math.sqrt(n);
        for (int i = 2; i <= limit; i++) {
            if (n % i == 0) return false;
        }
        return true;
    }
}""",
    "p8_n_square.java": """public class PerfectSquareCheck {
    public static boolean isPerfectSquare(int n) {
        if (n < 0) return false;
        int root = (int) Math.sqrt(n);
        return root * root == n;
    }
}""",
}

# P9: Matrix Multiplication
P9_FILES = {
    "p9_base_matmul.java": """public class MatrixMultiplication {
    public static int[][] multiply(int[][] A, int[][] B) {
        int r1 = A.length, c1 = A[0].length, c2 = B[0].length;
        int[][] C = new int[r1][c2];
        for (int i = 0; i < r1; i++) {
            for (int j = 0; j < c2; j++) {
                for (int k = 0; k < c1; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }
}""",
    "p9_v_rename.java": """public class MatMulOps {
    // Nhan 2 ma tran so nguyen 2 chieu
    public static int[][] multiply(int[][] firstMat, int[][] secondMat) {
        int rowsA = firstMat.length, colsA = firstMat[0].length, colsB = secondMat[0].length;
        int[][] res = new int[rowsA][colsB];
        for (int r = 0; r < rowsA; r++) {
            for (int c = 0; c < colsB; c++) {
                for (int p = 0; p < colsA; p++) {
                    res[r][c] += firstMat[r][p] * secondMat[p][c];
                }
            }
        }
        return res;
    }
}""",
    "p9_v_types.java": """public class MatrixMultiplicationDouble {
    public static double[][] multiply(double[][] A, double[][] B) {
        int r1 = A.length, c1 = A[0].length, c2 = B[0].length;
        double[][] C = new double[r1][c2];
        for (int i = 0; i < r1; i++) {
            for (int j = 0; j < c2; j++) {
                for (int k = 0; k < c1; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }
}""",
    "p9_v_while.java": """public class MatrixMulWhile {
    public static int[][] multiply(int[][] A, int[][] B) {
        int r1 = A.length, c1 = A[0].length, c2 = B[0].length;
        int[][] C = new int[r1][c2];
        int i = 0;
        while (i < r1) {
            int j = 0;
            while (j < c2) {
                int k = 0;
                while (k < c1) {
                    C[i][j] += A[i][k] * B[k][j];
                    k++;
                }
                j++;
            }
            i++;
        }
        return C;
    }
}""",
    "p9_v_flat.java": """public class FlatMatrixMultiplication {
    public static int[] multiply(int[] A, int[] B, int r1, int c1, int c2) {
        int[] C = new int[r1 * c2];
        for (int i = 0; i < r1; i++) {
            for (int j = 0; j < c2; j++) {
                int sum = 0;
                for (int k = 0; k < c1; k++) {
                    sum += A[i * c1 + k] * B[k * c2 + j];
                }
                C[i * c2 + j] = sum;
            }
        }
        return C;
    }
}""",
    "p9_n_transpose.java": """public class MatrixTranspose {
    public static int[][] transpose(int[][] A) {
        int rows = A.length, cols = A[0].length;
        int[][] T = new int[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                T[j][i] = A[i][j];
            }
        }
        return T;
    }
}""",
}

# P10: String Anagram
P10_FILES = {
    "p10_base_anagram.java": """import java.util.Arrays;

public class AnagramCheck {
    public static boolean isAnagram(String s1, String s2) {
        if (s1.length() != s2.length()) return false;
        char[] a1 = s1.toCharArray();
        char[] a2 = s2.toCharArray();
        Arrays.sort(a1);
        Arrays.sort(a2);
        return Arrays.equals(a1, a2);
    }
}""",
    "p10_v_rename.java": """import java.util.Arrays;

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
}""",
    "p10_v_types.java": """import java.util.Arrays;

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
}""",
    "p10_v_clean.java": """import java.util.Arrays;

public class AnagramCleaned {
    public static boolean isAnagram(String s1, String s2) {
        String c1 = s1.replaceAll("\\\\s+", "");
        String c2 = s2.replaceAll("\\\\s+", "");
        if (c1.length() != c2.length()) return false;
        char[] a = c1.toCharArray();
        char[] b = c2.toCharArray();
        Arrays.sort(a);
        Arrays.sort(b);
        return Arrays.equals(a, b);
    }
}""",
    "p10_v_frequency.java": """public class AnagramFrequency {
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
}""",
    "p10_n_substring.java": """public class SubstringCheck {
    public static boolean isSubstring(String s1, String s2) {
        return s1.contains(s2);
    }
}""",
}

# P11: Maximum Subarray Sum (Kadane)
P11_FILES = {
    "p11_base_kadane.java": """public class MaxSubarrayKadane {
    public static int maxSubArray(int[] nums) {
        int maxSoFar = nums[0];
        int currMax = nums[0];
        for (int i = 1; i < nums.length; i++) {
            currMax = Math.max(nums[i], currMax + nums[i]);
            maxSoFar = Math.max(maxSoFar, currMax);
        }
        return maxSoFar;
    }
}""",
    "p11_v_rename.java": """public class KadaneAlgorithm {
    // Tim tong day con lien tiep lon nhat
    public static int maxSubArray(int[] numbers) {
        int bestSum = numbers[0];
        int runningSum = numbers[0];
        for (int idx = 1; idx < numbers.length; idx++) {
            runningSum = Math.max(numbers[idx], runningSum + numbers[idx]);
            bestSum = Math.max(bestSum, runningSum);
        }
        return bestSum;
    }
}""",
    "p11_v_types.java": """public class MaxSubarrayLong {
    public static long maxSubArray(long[] nums) {
        long maxSoFar = nums[0];
        long currMax = nums[0];
        for (int i = 1; i < nums.length; i++) {
            currMax = Math.max(nums[i], currMax + nums[i]);
            maxSoFar = Math.max(maxSoFar, currMax);
        }
        return maxSoFar;
    }
}""",
    "p11_v_while.java": """public class MaxSubarrayWhile {
    public static int maxSubArray(int[] nums) {
        int maxSoFar = nums[0];
        int currMax = nums[0];
        int i = 1;
        while (i < nums.length) {
            if (currMax < 0) currMax = nums[i];
            else currMax += nums[i];
            if (currMax > maxSoFar) maxSoFar = currMax;
            i++;
        }
        return maxSoFar;
    }
}""",
    "p11_v_divide_conquer.java": """public class MaxSubarrayDivideConquer {
    public static int maxSubArray(int[] nums) {
        return solve(nums, 0, nums.length - 1);
    }
    private static int solve(int[] nums, int l, int r) {
        if (l == r) return nums[l];
        int m = l + (r - l) / 2;
        int leftMax = solve(nums, l, m);
        int rightMax = solve(nums, m + 1, r);
        int crossMax = crossSum(nums, l, m, r);
        return Math.max(Math.max(leftMax, rightMax), crossMax);
    }
    private static int crossSum(int[] nums, int l, int m, int r) {
        int leftSum = Integer.MIN_VALUE, sum = 0;
        for (int i = m; i >= l; i--) { sum += nums[i]; leftSum = Math.max(leftSum, sum); }
        int rightSum = Integer.MIN_VALUE; sum = 0;
        for (int i = m + 1; i <= r; i++) { sum += nums[i]; rightSum = Math.max(rightSum, sum); }
        return leftSum + rightSum;
    }
}""",
    "p11_n_min_subarray.java": """public class MinSubarray {
    public static int minSubArray(int[] nums) {
        int minSoFar = nums[0];
        int currMin = nums[0];
        for (int i = 1; i < nums.length; i++) {
            currMin = Math.min(nums[i], currMin + nums[i]);
            minSoFar = Math.min(minSoFar, currMin);
        }
        return minSoFar;
    }
}""",
}

# P12: Merge Sorted Arrays
P12_FILES = {
    "p12_base_merge.java": """public class MergeSortedArrays {
    public static int[] merge(int[] a, int[] b) {
        int[] res = new int[a.length + b.length];
        int i = 0, j = 0, k = 0;
        while (i < a.length && j < b.length) {
            if (a[i] <= b[j]) res[k++] = a[i++];
            else res[k++] = b[j++];
        }
        while (i < a.length) res[k++] = a[i++];
        while (j < b.length) res[k++] = b[j++];
        return res;
    }
}""",
    "p12_v_rename.java": """public class ArrayMerger {
    // Tron 2 mang da sap xep thanh mang moi
    public static int[] merge(int[] leftArr, int[] rightArr) {
        int[] merged = new int[leftArr.length + rightArr.length];
        int p1 = 0, p2 = 0, idx = 0;
        while (p1 < leftArr.length && p2 < rightArr.length) {
            if (leftArr[p1] <= rightArr[p2]) merged[idx++] = leftArr[p1++];
            else merged[idx++] = rightArr[p2++];
        }
        while (p1 < leftArr.length) merged[idx++] = leftArr[p1++];
        while (p2 < rightArr.length) merged[idx++] = rightArr[p2++];
        return merged;
    }
}""",
    "p12_v_types.java": """public class MergeSortedLongs {
    public static long[] merge(long[] a, long[] b) {
        long[] res = new long[a.length + b.length];
        int i = 0, j = 0, k = 0;
        while (i < a.length && j < b.length) {
            if (a[i] <= b[j]) res[k++] = a[i++];
            else res[k++] = b[j++];
        }
        while (i < a.length) res[k++] = a[i++];
        while (j < b.length) res[k++] = b[j++];
        return res;
    }
}""",
    "p12_v_while.java": """public class MergeSortedFor {
    public static int[] merge(int[] a, int[] b) {
        int n = a.length, m = b.length;
        int[] res = new int[n + m];
        int i = 0, j = 0;
        for (int k = 0; k < n + m; k++) {
            if (i >= n) res[k] = b[j++];
            else if (j >= m) res[k] = a[i++];
            else if (a[i] <= b[j]) res[k] = a[i++];
            else res[k] = b[j++];
        }
        return res;
    }
}""",
    "p12_v_queue.java": """import java.util.PriorityQueue;

public class MergeWithHeap {
    public static int[] merge(int[] a, int[] b) {
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int x : a) pq.add(x);
        for (int y : b) pq.add(y);
        int[] res = new int[a.length + b.length];
        int idx = 0;
        while (!pq.isEmpty()) res[idx++] = pq.poll();
        return res;
    }
}""",
    "p12_n_intersection.java": """import java.util.ArrayList;
import java.util.List;

public class ArrayIntersection {
    public static int[] intersect(int[] a, int[] b) {
        List<Integer> common = new ArrayList<>();
        int i = 0, j = 0;
        while (i < a.length && j < b.length) {
            if (a[i] == b[j]) { common.add(a[i]); i++; j++; }
            else if (a[i] < b[j]) i++;
            else j++;
        }
        int[] res = new int[common.size()];
        for (int k = 0; k < res.length; k++) res[k] = common.get(k);
        return res;
    }
}""",
}

# P13: Queue Data Structure
P13_FILES = {
    "p13_base_queue.java": """public class ArrayQueue {
    private int[] data = new int[100];
    private int front = 0, rear = 0, size = 0;
    public void enqueue(int x) { data[rear] = x; rear = (rear + 1) % data.length; size++; }
    public int dequeue() { int val = data[front]; front = (front + 1) % data.length; size--; return val; }
    public boolean isEmpty() { return size == 0; }
}""",
    "p13_v_rename.java": """public class CircularQueue {
    // Hang doi vong cai dat bang mang tinh
    private int[] buffer = new int[100];
    private int head = 0, tail = 0, count = 0;
    public void enqueue(int item) { buffer[tail] = item; tail = (tail + 1) % buffer.length; count++; }
    public int dequeue() { int item = buffer[head]; head = (head + 1) % buffer.length; count--; return item; }
    public boolean isEmpty() { return count == 0; }
}""",
    "p13_v_types.java": """public class GenericArrayQueue<T> {
    private Object[] data = new Object[100];
    private int front = 0, rear = 0, size = 0;
    public void enqueue(T x) { data[rear] = x; rear = (rear + 1) % data.length; size++; }
    @SuppressWarnings("unchecked")
    public T dequeue() { T val = (T) data[front]; front = (front + 1) % data.length; size--; return val; }
    public boolean isEmpty() { return size == 0; }
}""",
    "p13_v_while.java": """public class ResizableQueue {
    private int[] data = new int[4];
    private int front = 0, rear = 0, size = 0;
    public void enqueue(int x) {
        if (size == data.length) resize();
        data[rear] = x; rear = (rear + 1) % data.length; size++;
    }
    public int dequeue() { int val = data[front]; front = (front + 1) % data.length; size--; return val; }
    private void resize() {
        int[] next = new int[data.length * 2];
        for (int i = 0; i < size; i++) next[i] = data[(front + i) % data.length];
        data = next; front = 0; rear = size;
    }
    public boolean isEmpty() { return size == 0; }
}""",
    "p13_v_linked_queue.java": """public class LinkedQueue {
    private static class Node { int val; Node next; Node(int v) { val = v; } }
    private Node head, tail;
    public void enqueue(int x) {
        Node n = new Node(x);
        if (tail == null) { head = tail = n; }
        else { tail.next = n; tail = n; }
    }
    public int dequeue() {
        int val = head.val;
        head = head.next;
        if (head == null) tail = null;
        return val;
    }
    public boolean isEmpty() { return head == null; }
}""",
    "p13_n_stack.java": """public class ArrayStack {
    private int[] data = new int[100];
    private int top = -1;
    public void enqueue(int x) { data[++top] = x; } // goi enqueue nhung hanh vi la push
    public int dequeue() { return data[top--]; }     // hanh vi la pop
    public boolean isEmpty() { return top == -1; }
}""",
}

# P14: Power Function (x^n)
P14_FILES = {
    "p14_base_power.java": """public class PowerFunction {
    public static double power(double x, int n) {
        if (n == 0) return 1.0;
        long N = n;
        if (N < 0) { x = 1 / x; N = -N; }
        double ans = 1.0;
        for (long i = 0; i < N; i++) {
            ans *= x;
        }
        return ans;
    }
}""",
    "p14_v_rename.java": """public class PowerCalculator {
    // Tinh luy thua x mu n bang vong lap
    public static double power(double base, int exponent) {
        if (exponent == 0) return 1.0;
        long exp = exponent;
        if (exp < 0) { base = 1 / base; exp = -exp; }
        double result = 1.0;
        for (long step = 0; step < exp; step++) {
            result *= base;
        }
        return result;
    }
}""",
    "p14_v_types.java": """public class IntPower {
    public static long power(long base, int exp) {
        long res = 1L;
        for (int i = 0; i < exp; i++) {
            res *= base;
        }
        return res;
    }
}""",
    "p14_v_while.java": """public class PowerWhile {
    public static double power(double x, int n) {
        if (n == 0) return 1.0;
        long N = n;
        if (N < 0) { x = 1.0 / x; N = -N; }
        double ans = 1.0;
        while (N > 0) {
            ans *= x;
            N--;
        }
        return ans;
    }
}""",
    "p14_v_binary_exp.java": """public class BinaryExponentiation {
    public static double power(double x, int n) {
        long N = n;
        if (N < 0) { x = 1 / x; N = -N; }
        double ans = 1.0;
        double current = x;
        for (long i = N; i > 0; i /= 2) {
            if (i % 2 == 1) ans *= current;
            current *= current;
        }
        return ans;
    }
}""",
    "p14_n_sqrt.java": """public class NewtonSqrt {
    public static double power(double x, int dummy) { // tinh can bac hai
        if (x < 0) return Double.NaN;
        double guess = x / 2.0;
        for (int i = 0; i < 20; i++) {
            guess = (guess + x / guess) / 2.0;
        }
        return guess;
    }
}""",
}

# P15: Longest Common Prefix
P15_FILES = {
    "p15_base_lcp.java": """public class LongestCommonPrefix {
    public static String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) return "";
        String prefix = strs[0];
        for (int i = 1; i < strs.length; i++) {
            while (strs[i].indexOf(prefix) != 0) {
                prefix = prefix.substring(0, prefix.length() - 1);
                if (prefix.isEmpty()) return "";
            }
        }
        return prefix;
    }
}""",
    "p15_v_rename.java": """public class LcpSolver {
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
}""",
    "p15_v_types.java": """public class LcpCaseSensitive {
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
}""",
    "p15_v_while.java": """public class LcpWhile {
    public static String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) return "";
        String prefix = strs[0];
        int i = 1;
        while (i < strs.length) {
            while (strs[i].indexOf(prefix) != 0) {
                prefix = prefix.substring(0, prefix.length() - 1);
                if (prefix.length() == 0) return "";
            }
            i++;
        }
        return prefix;
    }
}""",
    "p15_v_vertical.java": """public class VerticalLcp {
    public static String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) return "";
        for (int i = 0; i < strs[0].length(); i++) {
            char c = strs[0].charAt(i);
            for (int j = 1; j < strs.length; j++) {
                if (i == strs[j].length() || strs[j].charAt(i) != c) {
                    return strs[0].substring(0, i);
                }
            }
        }
        return strs[0];
    }
}""",
    "p15_n_longest_string.java": """public class LongestStringFinder {
    public static String longestCommonPrefix(String[] strs) { // tim chuoi dai nhat
        if (strs == null || strs.length == 0) return "";
        String longest = strs[0];
        for (int i = 1; i < strs.length; i++) {
            if (strs[i].length() > longest.length()) longest = strs[i];
        }
        return longest;
    }
}""",
}

ALL_NEW_DICTS = [
    P6_FILES, P7_FILES, P8_FILES, P9_FILES, P10_FILES,
    P11_FILES, P12_FILES, P13_FILES, P14_FILES, P15_FILES,
]


def write_all_files():
    for d in ALL_NEW_DICTS:
        for fname, content in d.items():
            path = DATA_DIR / fname
            path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Tong so file Java trong {DATA_DIR}: {len(list(DATA_DIR.glob('*.java')))}")


def generate_manifest_1_to_5():
    # 15 problems
    problems = [f"p{i}" for i in range(1, 16)]
    clones = []
    nonclone_traps = []
    nonclone_cross = []

    # Map problem prefix to its files
    prob_files: dict[str, dict[str, str]] = {}
    for p in problems:
        flist = list(DATA_DIR.glob(f"{p}_*.java"))
        base = next(f for f in flist if "_base_" in f.name).name
        v_t1 = next(f for f in flist if "_v_rename" in f.name).name
        v_t2 = next(f for f in flist if "_v_types" in f.name or "_v_collections" in f.name or "_v_case" in f.name).name
        v_t3 = next(f for f in flist if "_v_while" in f.name or "_v_recursion" in f.name or "_v_levels" in f.name or "_v_clean" in f.name).name
        # v_t4 is semantic
        v_t4 = next(f for f in flist if f.name not in [base, v_t1, v_t2, v_t3] and "_n_" not in f.name).name
        n_hard = next(f for f in flist if "_n_" in f.name).name
        prob_files[p] = {
            "base": base,
            "v_t1": v_t1,
            "v_t2": v_t2,
            "v_t3": v_t3,
            "v_t4": v_t4,
            "n_hard": n_hard,
        }

    # Sinh toan bo cap clone
    for p, fs in prob_files.items():
        # Type-1
        clones.append({"pair_id": f"{p}_t1", "a": fs["base"], "b": fs["v_t1"], "label": 1, "category": "clone_type1", "note": f"{p} rename"})
        # Type-2
        clones.append({"pair_id": f"{p}_t2", "a": fs["base"], "b": fs["v_t2"], "label": 1, "category": "clone_type2", "note": f"{p} types"})
        clones.append({"pair_id": f"{p}_t1_t2", "a": fs["v_t1"], "b": fs["v_t2"], "label": 1, "category": "clone_type2", "note": f"{p} t1 vs t2"})
        # Type-3
        clones.append({"pair_id": f"{p}_t3", "a": fs["base"], "b": fs["v_t3"], "label": 1, "category": "clone_type3", "note": f"{p} control"})
        clones.append({"pair_id": f"{p}_t1_t3", "a": fs["v_t1"], "b": fs["v_t3"], "label": 1, "category": "clone_type3", "note": f"{p} t1 vs t3"})
        # Type-4
        clones.append({"pair_id": f"{p}_t4", "a": fs["base"], "b": fs["v_t4"], "label": 1, "category": "clone_type4", "note": f"{p} semantic different algorithm"})
        clones.append({"pair_id": f"{p}_t2_t4", "a": fs["v_t2"], "b": fs["v_t4"], "label": 1, "category": "clone_type4", "note": f"{p} t2 vs t4"})
        clones.append({"pair_id": f"{p}_t3_t4", "a": fs["v_t3"], "b": fs["v_t4"], "label": 1, "category": "clone_type4", "note": f"{p} t3 vs t4"})

    # Sinh non-clone trap (cung skeleton)
    for p, fs in prob_files.items():
        for variant_key in ["base", "v_t1", "v_t2", "v_t3", "v_t4"]:
            nonclone_traps.append({
                "pair_id": f"{p}_trap_{variant_key}",
                "a": fs[variant_key],
                "b": fs["n_hard"],
                "label": 0,
                "category": "nonclone_trap",
                "note": f"{p} {variant_key} vs skeleton trap",
            })

    # Sinh non-clone cross (khac bai)
    for i in range(len(problems)):
        for j in range(i + 1, len(problems)):
            pi, pj = problems[i], problems[j]
            fi, fj = prob_files[pi], prob_files[pj]
            nonclone_cross.append({"pair_id": f"X_{pi}{pj}_bb", "a": fi["base"], "b": fj["base"], "label": 0, "category": "nonclone_cross", "note": f"Cross {pi} vs {pj}"})
            nonclone_cross.append({"pair_id": f"X_{pi}{pj}_t1", "a": fi["v_t1"], "b": fj["v_t1"], "label": 0, "category": "nonclone_cross", "note": f"Cross {pi} vs {pj}"})
            nonclone_cross.append({"pair_id": f"X_{pi}{pj}_t2", "a": fi["v_t2"], "b": fj["v_t2"], "label": 0, "category": "nonclone_cross", "note": f"Cross {pi} vs {pj}"})
            nonclone_cross.append({"pair_id": f"X_{pi}{pj}_t3", "a": fi["v_t3"], "b": fj["v_t3"], "label": 0, "category": "nonclone_cross", "note": f"Cross {pi} vs {pj}"})
            nonclone_cross.append({"pair_id": f"X_{pi}{pj}_t4", "a": fi["v_t4"], "b": fj["v_t4"], "label": 0, "category": "nonclone_cross", "note": f"Cross {pi} vs {pj}"})
            nonclone_cross.append({"pair_id": f"X_{pi}{pj}_nh", "a": fi["n_hard"], "b": fj["n_hard"], "label": 0, "category": "nonclone_cross", "note": f"Cross {pi} vs {pj}"})

    random.seed(42)
    # Target: EXACT 120 Clones : 600 Non-clones (Total = 720, Ratio 1:5)
    random.shuffle(clones)
    selected_clones = clones[:120]

    random.shuffle(nonclone_traps)
    random.shuffle(nonclone_cross)
    # Lay 75 traps va 525 cross = 600 non-clones
    selected_traps = nonclone_traps[:75]
    selected_cross = nonclone_cross[:525]
    selected_nonclones = selected_traps + selected_cross

    final_pairs = selected_clones + selected_nonclones
    random.shuffle(final_pairs)

    manifest_path = DATA_DIR / "pairs.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for p in final_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Manifest {manifest_path}:")
    print(f"  - Tong so cap: {len(final_pairs)}")
    print(f"  - Clones (label 1): {len(selected_clones)}")
    print(f"  - Non-clones (label 0): {len(selected_nonclones)}")
    print(f"  - Ti le Clone : Non-clone = {len(selected_clones)} : {len(selected_nonclones)} = 1 : {len(selected_nonclones)//len(selected_clones)}")


def main():
    write_all_files()
    generate_manifest_1_to_5()


if __name__ == "__main__":
    main()
