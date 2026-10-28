public class MatMulOps {
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
}
