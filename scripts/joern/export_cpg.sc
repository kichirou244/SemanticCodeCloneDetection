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

  // Tên project phải **khác nhau giữa các lô**. Joern lưu project vào DB riêng
  // (`~/.joern`), còn các lô chạy tuần tự trong cùng một session — đặt cứng một
  // tên là lô thứ 2 trở đi sẽ đụng dữ liệu của lô trước, và lỗi đó chỉ lộ ra
  // khi chạy tới hàm thứ 500 chứ không phải ở lô 20. Lấy tên từ `srcDir` (đã khác
  // nhau theo lô rồi) để không phải thêm tham số.
  val projectName = "cpg_" + srcDir.replaceAll("[^A-Za-z0-9]", "_").takeRight(40)
  importCode(inputPath = srcDir, projectName = projectName)
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

  // Chọn **đúng một** method cho mỗi file, và chỉ ghi sau khi chọn xong.
  //
  // Joern sinh thêm method giả cho mỗi file: `<global>`, `<includes>`,
  // `<operator>.addition`… Chúng có `file.name` **TRÙNG** với file thật — smoke in
  // ra cả `classify` lẫn `<global>`, đều `List(smoke.cpp)`. Mà `sid` lấy từ tên
  // file, nên mọi method của một file đều ghi vào cùng một `{sid}.json` và cái
  // sau **đè** cái trước.
  //
  // Tệ hơn: `n_ok` đếm "file có tồn tại" nên vẫn đủ, và mọi gate đều xanh — hỏng
  // dữ liệu hoàn toàn im lặng. Đo được trên lô 500 hàm: 100 file có >1 method.
  // Chọn method THẬT (tên không bắt đầu bằng `<`); nếu còn nhiều thì lấy AST lớn
  // nhất, và nếu vẫn bằng thì so tên để không phụ thuộc thứ tự traversal.
  val bestReal = mutable.Map.empty[String, (Int, String, String, String)]
  val bestAny = mutable.Map.empty[String, (Int, String, String, String)]
  val hits = mutable.Map.empty[String, Int]
  var n_unindexed = 0

  for (m <- cpg.method.l) {
    val fname = m.file.name.l.headOption.getOrElse("")
    // File không có trong `index.tsv` là file giả của Joern (`<includes>`…), không
    // thuộc dataset. Bỏ qua thay vì đặt tên `<includes>` -> `_includes_.json` rồi
    // ghi ra file rác mà không ai hỏi. Nếu file THẬT lọt vào nhánh này thì gate
    // `n_missing` sẽ báo, chứ không im lặng nuốt.
    if (!fileToSample.contains(fname)) {
      n_unindexed += 1
    } else {
    val sid = fileToSample(fname)
    hits(sid) = hits.getOrElse(sid, 0) + 1
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
        // `.get` trả `Option`, không phải `Integer`. Viết `na == null` thì luôn
        // false (Option không bao giờ null) và `na.intValue` không tồn tại →
        // E008 compile, cả lô chết. Phải match Option.
        val naStr = nArgsByCall.get(n) match {
          case Some(v) => s""","n_args": $v"""
          case None    => ""
        }
        s"""{"id": $i, "label": ${esc(n.label)}$naStr}"""
      }.mkString(",")

      val body =
        s"""{"sample_id": ${esc(sid)}, "nodes": [$nodeRows], "edges": [""" +
          uniq.map(e => s"""{"src": ${e._1}, "dst": ${e._2}, "label": ${esc(e._3)}}""").mkString(",") +
          s"""]}"""

      // `.meta`: chẩn đoán từng hàm. `edge_types_missing` là phần quan trọng —
      // nó nói thẳng CPG này thiếu loại cạnh nào thay vì để ta phát hiện sau khi
      // train xong. `file_name` để Python đếm "file nào chứa nhiều hàm": cách v1
      // định nghĩa việc đó bằng `cpg.file.name(fname).method.l.size` — gọi step
      // có tham số, chưa từng được xác nhận là hợp lệ, nên bỏ; Python tự gom từ
      // `file_name` thay vì đoán trong Scala.
      val missing = List("AST", "CFG", "CALL", "REF", "ARGUMENT", "TARG",
                         "PARAMETER_OUT", "REACHING_DEF", "EVAL_ORDER")
        .filter(k => !uniq.exists(_._3 == k)).mkString(";")
      // Histogram PHẢI tính riêng cho hàm này. `edgeHist` ở trên là map cộng dồn
      // toàn lô (dùng cho dòng tổng kết cuối cùng); nếu ghi nó vào `.meta` thì hàm
      // đầu lô đúng còn hàm sau mang sẵn loại cạnh của hàm trước — và Python sẽ
      // đếm `edge_types_present` theo đúng cái bịa đó, tức một hàm thực sự thiếu
      // CFG vẫn bị báo là có CFG. `mapValues` + `toList` để chạy được cả 2.12/2.13.
      val hist = uniq.groupBy(_._3).mapValues(_.size).toList
        .sortBy(_._1).map { case (k, v) => s"$k=$v" }.mkString(";")
      val metaStr =
        (s"n_nodes\t${ast.size}\n" +
         s"n_edges\t${uniq.size}\n" +
         s"file_name\t${fname}\n" +
         s"method_name\t${m.name}\n" +
         s"method_signature\t${m.signature}\n" +
         s"edge_types\t$hist\n" +
         s"edge_types_missing\t$missing\n")

      val payload = (ast.size, m.name, body, metaStr)
      // Hơn nữa: khi AST bằng nhau thì so tên method. `cpg.method.l` không bảo đảm
      // thứ tự, nên nếu chỉ so `ast.size` thì kết quả có thể đổi giữa hai lần
      // chạy cùng dữ liệu — và ta không có cách nào phát hiện ngoài việc diff.
      def better(cand: (Int, String, String, String),
                 prev: (Int, String, String, String)): Boolean =
        if (cand._1 != prev._1) cand._1 > prev._1 else cand._2 < prev._2
      val target = if (!m.name.startsWith("<")) bestReal else bestAny
      target.get(sid) match {
        case Some(prev) if !better(payload, prev) => ()
        case _ => target(sid) = payload
      }
    }
    }
  }

  // Chỉ ghi những gì đã chọn. `bestReal` thắng `bestAny`; trong mỗi nhóm đã lưu
  // method có AST lớn nhất nên không còn phụ thuộc thứ tự `cpg.method.l`.
  val chosen = bestReal ++ bestAny.filter { case (k, _) => !bestReal.contains(k) }
  for ((sid, (_, _, body, metaStr)) <- chosen.toList.sortBy(_._1)) {
    Files.write(Paths.get(outDir, s"$sid.json"), body.getBytes(StandardCharsets.UTF_8))
    Files.write(Paths.get(outDir, s"$sid.meta"), metaStr.getBytes(StandardCharsets.UTF_8))
    n_written += 1
  }

  println(s"[export_cpg] wrote=$n_written out=$outDir")
  // Ba số này là bằng chứng chống ghi đè âm thầm: nếu `files_multi` > 0 mà
  // `wrote` vẫn bằng số sample thì đã chọn đúng 1 method mỗi file.
  println(s"[export_cpg] file co >1 method = ${hits.count(_._2 > 1)} " +
    s"| method bi ao bo (khong indexed) = $n_unindexed")
  println("[export_cpg] edge histogram (toàn lô): " +
    edgeHist.toList.sortBy(_._1).map { case (k, v) => s"$k=$v" }.mkString(" "))
  println(s"[export_cpg] node labels (toàn lô): " +
    labelHist.toList.sortBy(_._1).map { case (k, v) => s"$k=$v" }.mkString(" "))
}
