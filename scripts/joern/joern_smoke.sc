// SMOKE TEST Joern — chạy trước exporter, tốn vài chục giây.
//
// VÌ SAO CẦN FILE NÀY: máy dự án không có Java/Joern nên exporter KHÔNG thể được
// kiểm chứng local. Thay vì đoán tên hàm CPGQL rồi chạy cả 500 hàm (lỗi compile
// lộ ra sau 5 phút), script này chạy trên MỘT file C nhỏ và in ra:
//   1. method / label / số node thực tế,
//   2. danh sách step CPGQL mà chính Joern tự khai báo (`cpg.help`).
// Dán output về lại thì tên step trong `export_cpg.sc` được chốt chắc chắn thay
// vì phải đoán.
//
// CLI (đã đối chiếu docs.joern.io/interpreter):
//   joern --script scripts/joern/joern_smoke.sc --param srcFile=x.cpp

@main def exec(srcFile: String) = {
  importCode(inputPath = srcFile, projectName = "cpg_smoke")

  println("=== METHODS ===")
  cpg.method.foreach(m => println(s"  ${m.name} | ${m.signature}"))

  println("=== NODE LABELS ===")
  println("  " + cpg.all.label.l.distinct.mkString(", "))
  println(s"  n_nodes=${cpg.all.l.size} n_methods=${cpg.method.l.size}")

  println("=== EDGE STEP PROBES ===")
  // Từng dòng là một phép thử *độc lập*: script dừng ở dòng hỏng, các dòng trước
  // vẫn đã in ra, nên lần sửa sau vẫn hữu ích. Đây là cách "đo thay vì đoán".
  println("  ast=" + cpg.method.ast.l.size)
  println("  call=" + cpg.call.l.size)
  println("  argument=" + cpg.call.argument.l.size)
  println("  callOut=" + cpg.method.callOut.l.size)
  println("  ref=" + cpg.call.ref.l.size)
  println("  reachingDef=" + cpg.local.reachingDef.l.size)
  println("  methodReturn=" + cpg.method.methodReturn.l.size)
  println("  cfgNode=" + cpg.method.cfgNode.l.size)
  println("  parameter=" + cpg.method.parameter.l.size)

  println("=== CPGQL STEPS ===")
  println(cpg.help.toString)
}
