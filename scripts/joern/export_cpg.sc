// EXPORT CPG: Joern (CPG) → JSON theo format của ta.
//
// CLI:
//   joern --script scripts/joern/export_cpg.sc \
//          --param srcDir=work/batch_0 --param indexFile=work/batch_0/index.tsv \
//          --param outDir=artifacts/cpg_export
//
// THIẾT KẾ: **một lần khởi động Joern cho cả lô thư mục**, không phải một tiến
// trình cho mỗi hàm. JVM + scala-repl khởi động tốn ~10–30s; 62k tiến trình
// nghĩa là hàng tuần chỉ để khởi động JVM. Nên Python side gom `batch_size`
// file .cpp vào một thư mục rồi gọi script này đúng một lần cho lô đó.
//
// ============================ VÌ SAO VIẾT LẠI TỪ ĐẦU ============================
// Bản v1 chết ở bước Scala compile (0/20 rồi 0/500 hàm). Nguyên nhân đã được
// chứng minh bằng 34 lần quét từng step (`scripts/joern/api_scan.py`):
//   1. Type pattern `case x: io.joern.x2cpg.X2CpgNode` — package đó không tồn tại
//      trong Joern v4 (đã bác: không phải member).
//   2. `m.l`, `n.astChildren` gọi trực tiếp trên node đơn — CPGQL là ngôn ngữ
//      traversal, step nhận `Iterator[...]`, không nhận `Any`. Đúng dạng đã xác
//      nhận là `Iterator(n).astChildren`.
// Nên v2 bỏ hẳn mọi type pattern: chỉ dùng **thuộc tính của node lấy từ
// traversal** (`n.id`, `n.label` — đã xác nhận) và **step trên `Iterator(node)`**.
//
// Đã xác nhận bằng compiler (xem `docs/joern_api_v4.json`):
//   cpg.method / .ast / .call / .callOut / .callIn / .cfgNode / .parameter
//   cpg.call.argument, cpg.call.callee, method.name/fullName/signature/file.name
//   n.id, n.label, Iterator(n).astChildren, Iterator(n).astParent
// Còn suy ra (cùng cơ chế `Iterator(x).<step>`, sẽ xác nhận ở lần chạy kế):
//   Iterator(c).argument, Iterator(c).cfgNext
//
// ⚠ BẢN NÀY CỐ TÌNH CHƯA ĐỦ 9 CẠNH. Đã xác nhận chắc 4 loại (AST, CFG,
// ARGUMENT, PARAMETER_OUT); 5 loại còn lại chưa có đường đã kiểm chứng nên
// **không viết vào**: CALL, REF, TARG, REACHING_DEF, EVAL_ORDER. Chúng được liệt
// kê tường minh trong `.meta` (`edge_types_missing`) và trong stats của
// `graph_stats()` — im lặng nuốt cạnh thì không phân biệt được "mô hình học
// được CPG" với "mô hình học được nửa CPG".
// ============================================================================

@main def exec(srcDir: String, indexFile: String, outDir: String) = {
  import java.nio.charset.StandardCharsets
  import java.nio.file.{Files, Paths}
  import scala.collection.mutable

  // Bảng tên file -> sample_id (Python ghi sẵn; tên file trong Joern chỉ là *tên
  // file*, không có đường dẫn, nên phải ánh xạ ngược thay vì đoán).
  val src = scala.io.Source.fromFile(indexFile)
  val fileToSample = mutable.Map.empty[String, String]
  try {
    for (line <- src.getLines(); if line.nonEmpty) {
      val parts = line.split("\t")
      if (parts.length >= 2) fileToSample(parts(0)) = parts(1)
    }
  } finally src.close()

  importCode(inputPath = srcDir, projectName = "cpg_batch")
  Files.createDirectories(Paths.get(outDir))

  // Escape chuỗi JSON tối thiểu — không thêm dependency (JVM không có JSON sẵn).
  def esc(s: String): String = {
    val sb = new StringBuilder("\"")
    for (ch <- s) ch match {
      case '"'  => sb.append("\\\"")
      case '\\' => sb.append("\\\\")
      case '\n' => sb.append("\\n")
      case '\r' => sb.append("\\r")
      case '\t' => sb.append("\\t")
      case c if c < ' ' => sb.append("\\u%04x".format(c.toInt))
      case c    => sb.append(c)
    }
    sb.append('"').toString
  }

  var n_written = 0
  val edgeHist = mutable.Map.empty[String, Int]
  val labelHist = mutable.Map.empty[String, Int]

  for (m <- cpg.method.l) {
    val fname = m.file.name.l.headOption.getOrElse("")
    val sid = fileToSample.getOrElse(fname, fname.replaceAll("[^A-Za-z0-9_-]", "_"))
    val ast = m.ast.l.toList

    if (ast.nonEmpty) {
      // Id **cục bộ** = chỉ số trong `ast`, không dùng id nội bộ của Joern (trả
      // về giá trị ~1e11, dễ tràn int32 khi đưa vào PyG). Id chỉ là khoá liên kết
      // node↔edge trong JSON; `ingest.py` tự đánh số lại theo BFS.
      // IdentityHashMap: node Joern là case class nên `equals` là so sánh cấu
      // trúc — hai node trùng mọi trường sẽ bị coi là một, mất cạnh.
      val idx = new java.util.IdentityHashMap[Any, Integer]()
      ast.zipWithIndex.foreach { case (n, i) => idx.put(n, Integer.valueOf(i)) }

      val edges = mutable.ArrayBuffer.empty[(Int, Int, String)]
      def add(a: Any, b: Any, kind: String): Unit = {
        val s = idx.get(a); val d = idx.get(b)
        if (s != null && d != null && s.intValue != d.intValue) {
          edges += ((s.intValue, d.intValue, kind))
        }
      }

      ast.foreach { n => labelHist(n.label) = labelHist.getOrElse(n.label, 0) + 1 }

      // --- AST: cha -> con. `astChildren` là bước đi xuống cây cú pháp -------
      ast.foreach { n => Iterator(n).astChildren.l.foreach(c => add(n, c, "AST")) }

      // --- CFG: cạnh điều khiển thật, không phải "nối node liên tiếp" ----------
      // Chỉ node CFG mới có `cfgNext`; lấy qua `m.cfgNode` để không gọi nhầm lên
      // node AST thuần (CALL/LITERAL không phải CFG node).
      val cfg = m.cfgNode.l.toList
      cfg.foreach { c => Iterator(c).cfgNext.l.foreach(s => add(c, s, "CFG")) }

      // --- ARGUMENT + PARAMETER_OUT -----------------------------------------
      // `n_args` là feature duy nhất của node (FEATURE_DIM=1): số đối số đã
      // chuẩn hoá log1p — tín hiệu ngữ nghĩa mà node type không nắm được.
      val params = m.parameter.l.toSet
      val nArgsByCall = mutable.Map.empty[Any, Integer]
      m.call.l.toList.foreach { c =>
        val args = Iterator(c).argument.l.toList
        nArgsByCall.put(c, Integer.valueOf(args.size))
        args.foreach(a => add(c, a, "ARGUMENT"))
        args.foreach(a => if (params.contains(a)) add(c, a, "PARAMETER_OUT"))
      }

      // Gộp trùng rồi sắp xếp cho byte-for-byte ổn định giữa các lần chạy.
      val uniq = edges.distinct.sortBy(e => (e._1, e._2, e._3))
      uniq.foreach(e => edgeHist(e._3) = edgeHist.getOrElse(e._3, 0) + 1)

      val nodeRows = ast.zipWithIndex.map { case (n, i) =>
        val na = nArgsByCall.get(n)
        val naStr = if (na == null) "" else s""","n_args": ${na.intValue}"""
        s"""{"id": $i, "label": ${esc(n.label)}$naStr}"""
      }.mkString(",")

      val body =
        s"""{"sample_id": ${esc(sid)}, "nodes": [$nodeRows], "edges": [""" +
          uniq.map(e => s"""{"src": ${e._1}, "dst": ${e._2}, "label": ${esc(e._3)}}""").mkString(",") +
          s"""]}"""

      Files.write(Paths.get(outDir, s"$sid.json"), body.getBytes(StandardCharsets.UTF_8))

      // `.meta`: chẩn đoán từng hàm. `edge_types_missing` là phần quan trọng —
      // nó nói thẳng CPG này thiếu loại cạnh nào thay vì để ta phát hiện sau khi
      // train xong. `file_name` để Python đếm "file nào chứa nhiều hàm": cách v1
      // định nghĩa việc đó bằng `cpg.file.name(fname).method.l.size` — gọi step
      // có tham số, chưa từng được xác nhận là hợp lệ, nên bỏ; Python tự gom từ
      // `file_name` thay vì đoán trong Scala.
      val missing = List("AST", "CFG", "CALL", "REF", "ARGUMENT", "TARG",
                         "PARAMETER_OUT", "REACHING_DEF", "EVAL_ORDER")
        .filter(k => !uniq.exists(_._3 == k)).mkString(";")
      val hist = edgeHist.keys.toList.sorted.map(k => s"$k=${edgeHist(k)}").mkString(";")
      Files.write(Paths.get(outDir, s"$sid.meta"),
        (s"n_nodes\t${ast.size}\n" +
         s"n_edges\t${uniq.size}\n" +
         s"file_name\t${fname}\n" +
         s"method_name\t${m.name}\n" +
         s"method_signature\t${m.signature}\n" +
         s"edge_types\t$hist\n" +
         s"edge_types_missing\t$missing\n").getBytes(StandardCharsets.UTF_8))
      n_written += 1
    }
  }

  println(s"[export_cpg] wrote=$n_written out=$outDir")
  println("[export_cpg] edge histogram (toàn lô): " +
    edgeHist.toList.sortBy(_._1).map { case (k, v) => s"$k=$v" }.mkString(" "))
  println(s"[export_cpg] node labels (toàn lô): " +
    labelHist.toList.sortBy(_._1).map { case (k, v) => s"$k=$v" }.mkString(" "))
}
