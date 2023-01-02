/** @file has_restrict.cu
 * Test if the compiler supports given RESTRICT keyword
*/
int f(void * RESTRICT x);
int main() {return 0;}
