// PROBE API JOERN — script **không thể fail do đoán sai tên step**.
//
// VÌ SAO CÓ FILE NÀY: Joern v4 đổi tên class/step so với bản cũ. Ba lần chạy
// Kaggle liên tiếp đều chết ở bước *compile* trước khi in được gì:
//   1. `cpg.call.ref`      → không phải member trên Iterator[Call]
//   2. `cpg.local.reachingDef` → không phải member trên Iterator[Local]
//   3. `io.joern.x2cpg.X2CpgNode` → package không tồn tại trong v4
// Mỗi lần mất ~15 phút vòng Kaggle để đổi một tên. Nguyên nhân gốc: **CPGQL
// định nghĩa step trên `Iterator[T]`/`Traversal`, không phải trên node đơn**, nên
// `n.astChildren` (n là một node) là cú pháp sai dù tên step có thể đúng.
//
// CÁCH SỬA Ở ĐÂY: dùng **reflection** (`getClass.getMethods`) để hỏi Joern trực
// tiếp "node này có member nào", thay vì đoán. Reflection không cần biết trước
// tên class hay tên step nên **không có tên nào có thể sai** — đây là lý do file
// này chắc chắn compile được. Chỉ dùng những step đã được chạy 1 xác nhận:
// `cpg.method.ast`, `cpg.all.label`, `importCode`.
//
// Đầu ra (grep được):
//   CLASS <label> <tên đầy đủ>        → package thật của node type
//   API  <label> <tên> <số phần tử>   → member tồn tại và gọi được không
//   MISS <label> <tên>                 → không có member đó
//
// CLI: joern --script scripts/joern/joern_probe.sc --param srcFile=x.cpp

import scala.language.reflectiveCalls

@main def exec(srcFile: String) = {
  importCode(inputPath = srcFile, projectName = "cpg_probe")

  val nodes = cpg.all.l
  println(s"[probe] n_nodes=${nodes.size} n_methods=${cpg.method.l.size}")
  println(s"[probe] traversal_runtime_type=${cpg.all.getClass.getName}")

  // Một node đại diện cho mỗi nhãn — đủ để hỏi về member mà không spam 10^5 dòng.
  // `label` cũng đọc qua reflection: nếu `label` chỉ tồn tại dưới dạng step trên
  // traversal chứ không phải method của node thì `n.label` sẽ không compile.
  def lblOf(n: Any): String = {
    val m = n.getClass.getMethods.find(x => x.getName == "label" && x.getParameterCount == 0)
    if (m.isEmpty) "?" else m.get.invoke(n).toString
  }
  val byLabel = scala.collection.mutable.LinkedHashMap.empty[String, Any]
  for (n <- nodes) { val l = lblOf(n); if (!byLabel.contains(l)) byLabel(l) = n }

  println("=== CLASS (tên đầy đủ của từng node type) ===")
  for ((lbl, n) <- byLabel) println(s"CLASS $lbl ${n.getClass.getName}")

  // Đây là danh sách câu hỏi: mọi member mà `export_cpg.sc` cần dùng cho 9 loại
  // cạnh. Có `getMethods` nên không cần biết trước bất kỳ tên nào tồn tại.
  val wanted = List(
    // thuộc tính cơ bản — mọi node đều phải có
    "id", "label", "code", "name", "lineNumber", "offset", "order", "columnNumber",
    // AST
    "ast", "astChildren", "astParent", "parent", "children",
    // lời gọi hàm
    "callOut", "callIn", "callee", "argument", "methodFullName", "ref",
    // trả về / tham số
    "methodReturn", "parameter", "param",
    // data flow
    "reachingDef", "reachingDefinitions",
    // CFG
    "cfgNode", "cfgNext", "cfgPrev", "cfgFirst", "cfgLast", "next", "prev",
    // khác
    "file", "signature", "evaluation", "definedBy")

  println("=== API (member thật của Joern v4) ===")
  for ((lbl, n) <- byLabel) {
    val ms = n.getClass.getMethods
    val have = ms.map(_.getName).toSet
    for (w <- wanted) {
      if (have.contains(w)) {
        // Gọi thật qua reflection trong Try: có tên chưa chắc gọi được
        // (có thể cần tham số, có thể trả kiểu khác) — phân biệt "có" với "có
        // nhưng hỏng" là điều exporter cần biết.
        val r = try {
          val cand = ms.filter(m => m.getName == w && m.getParameterCount == 0)
          if (cand.isEmpty) "CO_THAM_SO_KHONG_GOI_NOC"
          else cand.head.invoke(n) match {
            case null            => "null"
            case it: Iterable[_] =>
              val l = it.toList
              val smp = l.take(2).map(x => x.getClass.getSimpleName).mkString("|")
              s"${l.size}pt mau=$smp"
            case other           => s"kieu=${other.getClass.getSimpleName}"
          }
        } catch {
          case e: Throwable => s"LOI ${e.getClass.getSimpleName}"
        }
        println(s"API  $lbl $w $r")
      } else {
        println(s"MISS $lbl $w")
      }
    }
  }

  // Liệt kê **mọi** member của node CALL và LOCAL — nơi cạnh REF và REACHING_DEF
  // nằm. `MISS` ở trên chỉ trả lời "ta đoán tên này, nó không có"; phần này trả
  // lời "thật ra Joern gọi nó là gì", tức là không còn cần đoán nữa.
  println("=== ALL MEMBERS: CALL ===")
  byLabel.get("CALL").foreach { n =>
    println(n.getClass.getMethods.map(_.getName).distinct.sorted.mkString(" "))
  }
  println("=== ALL MEMBERS: LOCAL ===")
  byLabel.get("LOCAL").foreach { n =>
    println(n.getClass.getMethods.map(_.getName).distinct.sorted.mkString(" "))
  }
  println("=== ALL MEMBERS: IDENTIFIER ===")
  byLabel.get("IDENTIFIER").foreach { n =>
    println(n.getClass.getMethods.map(_.getName).distinct.sorted.mkString(" "))
  }
  println(s"[probe] xong")
}
