// SMOKE TEST Joern — chạy trước exporter, tốn chưa tới 1 phút.
//
// VÌ SAO CẦN FILE NÀY: máy dự án không có Java/Joern nên exporter KHÔNG thể được
// kiểm chứng local. Và bài học từ lần chạy đầu: **Scala compile trước, chạy sau** —
// một dòng đặt tên step sai là mất trắng cả lần chạy, không in được gì. Nên
// script này chỉ dùng những gì đã biết chắc là hợp lệ, rồi **tự in ra danh sách
// step Joern khai báo** để tên step trong `export_cpg.sc` được chốt bằng bằng
// chứng thay vì đoán.
//
// Lần chạy 1 (Kaggle) đã xác nhận hợp lệ: `cpg.method.ast`, `cpg.call.argument`,
// `cpg.method.callOut`, `cpg.method.methodReturn`, `cpg.method.cfgNode`,
// `cpg.method.parameter`, `cpg.all.label`, `cpg.all.l.size`.
// Lần chạy 1 cũng đã bác bỏ: `cpg.call.ref` và `cpg.local.reachingDef` — không
// phải member trên Iterator[...] (tên đúng lấy từ các bảng `.help` in ra).
//
// CLI (đã đối chiếu docs.joern.io/interpreter):
//   joern --script scripts/joern/joern_smoke.sc --param srcFile=x.cpp

@main def exec(srcFile: String) = {
  importCode(inputPath = srcFile, projectName = "cpg_smoke")

  println("=== METHODS ===")
  cpg.method.foreach(m => println(s"  ${m.name} | ${m.signature} | ${m.file.name.l}"))

  println("=== NODE LABELS ===")
  println("  " + cpg.all.label.l.distinct.mkString(", "))
  println(s"  n_nodes=${cpg.all.l.size} n_methods=${cpg.method.l.size}")

  println("=== EDGE PROBES (đã biết chắc hợp lệ) ===")
  println("  ast=" + cpg.method.ast.l.size)
  println("  argument=" + cpg.call.argument.l.size)
  println("  callOut=" + cpg.method.callOut.l.size)
  println("  methodReturn=" + cpg.method.methodReturn.l.size)
  println("  cfgNode=" + cpg.method.cfgNode.l.size)
  println("  parameter=" + cpg.method.parameter.l.size)

  // Bảng step của Joern: đây là câu trả lời chính thức cho "tên step nào đúng",
  // in ra để chốt tên trong exporter. Cần nhất là `cpg.call` (tìm cạnh REF) và
  // `cpg.identifier` / `cpg.local` (tìm cạnh REACHING_DEF).
  println("=== HELP cpg.method ===")
  println(cpg.method.help.toString)
  println("=== HELP cpg.call ===")
  println(cpg.call.help.toString)
  println("=== HELP cpg.identifier ===")
  println(cpg.identifier.help.toString)
  println("=== HELP cpg.local ===")
  println(cpg.local.help.toString)
}
