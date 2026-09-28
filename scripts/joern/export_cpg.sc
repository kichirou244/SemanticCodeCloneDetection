// EXPORT CPG: Joern (CPG) → JSON theo format của ta.
//
// CLI (đã đối chiếu docs.joern.io/interpreter):
//   joern --script scripts/joern/export_cpg.sc \
//          --param srcDir=work/batch_0 --param indexFile=work/batch_0/index.tsv \
//          --param outDir=artifacts/cpg_export
//
// THIẾT KẾ: **một lần khởi động Joern cho cả lô thư mục**, không phải một tiến
// trình cho mỗi hàm. JVM + scala-repl khởi động tốn ~10–30s; 62k tiến trình
// nghĩa là hàng tuần chỉ để khởi động JVM. Nên Python side gom `batch_size`
// file .cpp vào một thư mục rồi gọi script này đúng một lần cho lô đó.
//
// ⚠ CHƯA KIỂM CHỨNG: tên step CPGQL dưới đây phải được xác nhận bằng
// `joern_smoke.sc` (in ra `cpg.help` + probe từng step) trước khi chạy lô lớn.
// Mỗi loại cạnh là một hàm riêng + một dòng probe độc lập: sửa một chỗ không
// phá các chỗ còn lại. Xem `scripts/build_cpg_export.py` (phần gọi Joern).
//
// HỢP ĐỒNG DỮ LIỆU (đóng băng cùng `src/representations/cpg/schema.py`):
//   - node: 13 nhãn trong schema; nhãn lạ → bỏ (đếm ở `.meta`).
//   - cạnh: 9 nhãn trong schema; nhãn lạ → bỏ.
//   - `METHOD` là node gốc; chỉ lấy phần đồ thị **bên trong method đó**, không
//     kéo cả file (kéo cả file làm graph phình to và lẫn giữa các hàm).
// Đổi contract này = đổi feature đầu vào mô hình ⇒ phải ghi decision log.

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}

val NodeKeep = Set("METHOD", "PARAMETER", "LOCAL", "FIELD", "CALL", "IDENTIFIER",
  "LITERAL", "OPERATOR", "CONTROL_STRUCTURE", "JUMP_TARGET", "BLOCK", "TYPE_DECL")

/** Escape chuỗi JSON tối thiểu — không thêm dependency (JVM không có JSON sẵn). */
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

def nodeId(n: Any): Long = n match {
  case x: io.joern.x2cpg.X2CpgNode => x.id
  case _ => -1L
}

/** Cạnh AST: cha -> con. `astChildren` là bước đi xuống trong cây cú pháp. */
def astEdges(nodes: List[Any]): List[(Long, Long, String)] =
  nodes.flatMap { n =>
    n match {
      case x: io.joern.x2cpg.X2CpgAstNode =>
        x.astChildren.map(c => (nodeId(x), nodeId(c), "AST")).toList
      case _ => Nil
    }
  }

/** Cạnh CALL: hàm/lệnh -> lời gọi mà nó thực hiện. */
def callEdges(nodes: List[Any]): List[(Long, Long, String)] =
  nodes.flatMap {
    case c: io.joern.x2cpg.X2CpgCall => c.callOut.map(t => (nodeId(c), nodeId(t), "CALL")).toList
    case _ => Nil
  }

/** Cạnh ARGUMENT: lời gọi -> đối số tương ứng (thứ tự = `argumentIndex`). */
def argEdges(nodes: List[Any]): List[(Long, Long, String)] =
  nodes.flatMap {
    case c: io.joern.x2cpg.X2CpgCall =>
      c.argument.map(a => (nodeId(c), nodeId(a), "ARGUMENT")).toList
    case _ => Nil
  }

/** Cạnh REF: node -> node được tham chiếu (tên hàm, biến toàn cục...). */
def refEdges(nodes: List[Any]): List[(Long, Long, String)] =
  nodes.flatMap {
    case c: io.joern.x2cpg.X2CpgCall => c.ref.map(r => (nodeId(c), nodeId(r), "REF")).toList
    case _ => Nil
  }

/** Cạnh REACHING_DEF: định danh -> định nghĩa có thể tới được nó (data flow). */
def reachingDefEdges(nodes: List[Any]): List[(Long, Long, String)] =
  nodes.flatMap {
    case u: io.joern.x2cpg.X2CpgIdentifier => u.reachingDef.map(d => (nodeId(u), nodeId(d), "REACHING_DEF")).toList
    case _ => Nil
  }

/**
 * Cạnh CFG: Joern không lưu "cạnh CFG" như một thuộc tính đơn giản, mà là chuỗi
 * node theo thứ tự thực thi. Vì vậy nối **node liên tiếp trong `cfgNode`** theo
 * `order` — đây là cách dựng CFG tiêu chuẩn từ CPG, và cũng là thứ mà GAT cần.
 */
def cfgEdges(method: Any, nodes: List[Any]): List[(Long, Long, String)] = {
  val order = nodes.map(n => (nodeId(n), n)).collect {
    case (id, c: io.joern.codepropertygraph.CfgNode) => (c.order, id)
  }.sortBy(_._1)
  order.sliding(2).collect { case List((_, a), (_, b)) => (a, b, "CFG") }.toList
}

@main def exec(srcDir: String, indexFile: String, outDir: String) = {
  // Bảng tên file -> sample_id (Python ghi sẵn; tên file trong Joern là *tên
  // file*, không có đường dẫn, nên phải ánh xạ ngược thay vì đoán).
  val fileToSample = scala.collection.mutable.Map.empty[String, String]
  for (line <- scala.io.Source.fromFile(indexFile).getLines(); if line.nonEmpty) {
    val parts = line.split("\t")
    if (parts.length >= 2) fileToSample(parts(0)) = parts(1)
  }

  importCode(inputPath = srcDir, projectName = "cpg_batch")
  Files.createDirectories(Paths.get(outDir))

  var n_written = 0
  for (m <- cpg.method.l) {
    val fname = m.file.name.l.headOption.getOrElse("")
    val sid = fileToSample.getOrElse(fname, fname.replaceAll("[^A-Za-z0-9_-]", "_"))
    val nodes = m.l                      // cây cú pháp bên trong method
    val kept = nodes.filter(n => NodeKeep.contains(n.label))

    val edges = (astEdges(kept) ++ callEdges(kept) ++ argEdges(kept) ++
                 refEdges(kept) ++ reachingDefEdges(kept) ++ cfgEdges(m, kept))
      .filter((s, d, _) => s != d)
      .distinct
      .sortBy(e => (e._1, e._2, e._3))

    if (kept.nonEmpty) {
      val body =
        s"""{"sample_id": ${esc(sid)}, "nodes": [""" +
          kept.map(n => s"""{"id": ${nodeId(n)}, "label": ${esc(n.label)}}""").mkString(",") +
          s"""], "edges": [""" +
          edges.map(e => s"""{"src": ${e._1}, "dst": ${e._2}, "label": ${esc(e._3)}}""").mkString(",") +
          s"""]}"""
      Files.write(Paths.get(outDir, s"$sid.json"), body.getBytes(StandardCharsets.UTF_8))
      Files.write(Paths.get(outDir, s"$sid.meta"),
        s"n_methods_in_file\t${cpg.file.name(fname).method.l.size}\nn_nodes\t${kept.size}\nn_edges\t${edges.size}\nmethod_name\t${m.name}\n"
          .getBytes(StandardCharsets.UTF_8))
      n_written += 1
    }
  }
  println(s"[export_cpg] wrote=$n_written out=$outDir")
}
