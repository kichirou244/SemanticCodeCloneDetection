# Bộ kiểm thử định tính Custom Suite cho Java (BigCloneBench)

Bộ kiểm thử được thiết kế riêng cho ngôn ngữ **Java** nhằm đánh giá định tính phản ứng của các mô hình phát hiện trùng lặp mã nguồn (**M1 CodeBERT, M2 GraphCodeBERT, M3 ASTNN, M4 CPG-GNN**) trước các biến thể cú pháp, cấu trúc và ngữ nghĩa trong thực tế.

## 1. Cấu trúc 5 bài toán chuẩn

| Bài | Tên bài toán | Base File | Biến thể Clone (T1 - T4) | Non-clone khó (Skeleton trap) |
| :--- | :--- | :--- | :--- | :--- |
| **P1** | Sắp xếp mảng | `p1_base_bubblesort.java` | `p1_v_rename.java` (T1)<br>`p1_v_types.java` (T2)<br>`p1_v_while.java` (T3)<br>`p1_v_quicksort.java` (T4) | `p1_n_reverse.java`<br>*(Cùng 2 vòng lặp & swap nhưng đảo ngược mảng)* |
| **P2** | Ước chung lớn nhất (GCD) | `p2_base_gcd.java` | `p2_v_rename.java` (T1)<br>`p2_v_types.java` (T2)<br>`p2_v_recursion.java` (T3)<br>`p2_v_binary_gcd.java` (T4) | `p2_n_digitsum.java`<br>*(Cùng vòng lặp while với `%` nhưng tính tổng chữ số)* |
| **P3** | Duyệt đồ thị (Reachability) | `p3_base_bfs.java` | `p3_v_rename.java` (T1)<br>`p3_v_collections.java` (T2)<br>`p3_v_levels.java` (T3)<br>`p3_v_dfs.java` (T4) | `p3_n_bipartite.java`<br>*(Cùng BFS duyệt kề nhưng kiểm tra đồ thị phân đôi)* |
| **P4** | Chuỗi đối xứng (Palindrome) | `p4_base_palindrome.java` | `p4_v_rename.java` (T1)<br>`p4_v_types.java` (T2)<br>`p4_v_clean.java` (T3)<br>`p4_v_reverse.java` (T4) | `p4_n_reverse_string.java`<br>*(Cùng 2 con trỏ swap ký tự nhưng tạo chuỗi đảo ngược)* |
| **P5** | Cấu trúc dữ liệu OOP (Linked List) | `p5_base_list.java` | `p5_v_rename.java` (T1)<br>`p5_v_types.java` (T2)<br>`p5_v_while.java` (T3)<br>`p5_v_dynamic_array.java` (T4) | `p5_n_stack.java`<br>*(Cùng class Node con trỏ nhưng cài đặt ngăn xếp LIFO Stack)* |

---

## 2. Phân loại cặp đánh giá (Tổng cộng: 80 cặp)

1. **Nhóm Trùng lặp (Clones - 35 cặp, label 1):**
   - **Type-1 (5 cặp):** Đổi tên biến, hàm, định dạng, thêm ghi chú comment.
   - **Type-2 (10 cặp):** Thay đổi kiểu dữ liệu (`int` $\rightarrow$ `long`, `Integer[]`, Generic `<T>`, Collection API).
   - **Type-3 (10 cặp):** Thay đổi cấu trúc điều khiển (`for` $\leftrightarrow$ `while`, đệ quy, thêm điều kiện dừng sớm, xử lý chuỗi chuẩn hóa).
   - **Type-4 Semantic Clones (10 cặp):** Cùng giải quyết trọn vẹn một bài toán nhưng sử dụng thuật toán hoàn toàn khác biệt:
     - BubbleSort vs QuickSort.
     - Euclid GCD vs Stein Binary GCD (dịch bit).
     - BFS Queue vs DFS Đệ quy.
     - 2 con trỏ Palindrome vs `StringBuilder.reverse()`.
     - SinglyLinkedList vs DynamicArrayList (cùng cung cấp `add`, `contains`, `sum`).

2. **Nhóm Không trùng lặp (Non-clones - 45 cặp, label 0):**
   - **Non-clone khó / Cạm bẫy cấu trúc (25 cặp):** Cùng khung sườn code (skeleton), cùng số lượng vòng lặp, biến hoặc cấu trúc dữ liệu con trỏ nhưng ngữ nghĩa bài toán khác nhau hoàn toàn.
   - **Non-clone liên bài toán (20 cặp):** Các cặp chéo giữa các bài toán khác nhau (Sorting vs Graph, Math vs Linked List,...).
