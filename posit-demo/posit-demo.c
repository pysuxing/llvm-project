/********************************************************
 * posit demo
 *
 * clang -S -emit-llvm posit-demo.c
 * 
 * The generated IR will be output to posit-demo.ll
 */

__posit16 posit16_demo() {
  int i = 1;
  __posit16 x = i, y = 0.5;
  y = x + y;
  return y;
}

__posit32 posit32_demo() {
  __posit32 x = 1, y = 0.5;
  y = x - y;
  return y;
}

__posit64 posit64_demo() {
  __posit64 x = 1, y = 0.5;
  y = x * y;
  return y;
}

__posit16_1 posit16_1_demo() {
  __posit16_1 x = 1, y = 0.5;
  y = x + y;
  return y;
}

__posit32_3 posit32_3_demo() {
  __posit32_3 x = 1, y = 0.5;
  y = x / y;
  return y;
}

// bfloat bfloat_demo() {
//   double t = 0;
//   bfloat x = .3, y = t;
//   return x * y;
// }

int main() {
  posit16_demo();
  posit32_demo();
  posit64_demo();
  posit32_3_demo();
  posit16_1_demo();
  // bfloat_demo();
  return 0;
}
