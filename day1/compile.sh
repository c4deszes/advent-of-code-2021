gcc -c program.S && ld program.o
cat data.txt | ./a.out | hexyl
# gdb -ex start --args ./a.out