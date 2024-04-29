;********************************************************
; posit demo
;
; clang -O2 -S -emit-llvm -mcpu=xiangjiang posit-demo.c
; 
; The generated IR will be output to posit-demo.ll
;*******************************************************/

; ModuleID = 'posit-demo/posit-demo.c'
source_filename = "posit-demo/posit-vector-demo.c"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef <4 x posit32> @posit32x4_demo(<4 x posit32> noundef %x, <4 x posit32> noundef %y, <4 x posit32> noundef %z) local_unnamed_addr #0 {
entry:
  %add = fadd <4 x posit32> %y, %z
  %add1 = fsub <4 x posit32> %z, %x
  %0 = fmul <4 x posit32> %add1, %y
  %1 = fdiv <4 x posit32> %x, %0
  %2 = fneg <4 x posit32> %1
  %3 = tail call <4 x posit32> @llvm.fmuladd.4xposit32(<4 x posit32> %add, <4 x posit32> %1, <4 x posit32> %0)
  %4 = tail call <4 x posit32> @llvm.fmuladd.4xposit32(<4 x posit32> %2, <4 x posit32> %1, <4 x posit32> %3)
  ret <4 x posit32> %4
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="xiangjiang" "target-features"="+bf16,+fp-armv8,+neon,+posit,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 19.0.0git (git@202.197.20.100:xsu/llvm-project.git 1386cfb627b1143ab0e99a162d02bb0c70125d83)"}
