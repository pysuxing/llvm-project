/********************************************************
 * posit demo
 *
 * clang -O2 -S -emit-llvm -mcpu=xiangjiang posit-demo.c
 * 
 * The generated IR will be output to posit-demo.ll
 ********************************************************/

// __posit16 posit16_demo() {
//   int i = 1;
//   __posit16 x = i, y = 0.5;
//   y = x + y;
//   return y;
// }

double posit32_demo() {
  __posit32 x = 1, y = 0.5, z = 2;
  x = y - 1;
  y += y;
  y = y / x;
  z += x * y;
  y -= z * x;
  double res = y;
  return res;
}

// __posit64 posit64_demo() {
//   __posit64 x = 1, y = 0.5;
//   y = x + y;
//   return y;
// }

// __posit16_1 posit16_1_demo() {
//   __posit16_1 x = 1, y = 0.5;
//   y = x + y;
//   return y;
// }

// __posit32_3 posit32_3_demo(__posit32_3 x, __posit32_3 y, __posit32_3 z) {
//   x = y + z;
//   y += y;
//   z += x * y;
//   return z;
// }

// __bf16 bfloat_demo(double t) {
//   __bf16 x = .3, y = t;
//   return x * y;
// }
