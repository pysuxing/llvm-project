#include<stdio.h>

posit32 simple_test(double a, double b){
  posit32 x = a;
  posit32 y = 3.0 * b;
  return x+y;
}

int main(){
	posit32 a = 0.323;
	printf("res a is %.20e, correct answer is 4.323\n",(double)a);
	a = simple_test(a, a);	
	printf("res a is %.20e, correct answer is 4.323\n",(double)a);
}
	
