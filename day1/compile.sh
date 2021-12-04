mkdir bin
gcc -Og -g -c program.S -o ./bin/program.o && ld ./bin/program.o -o ./bin/program
cat test.txt | ./bin/program | hexyl
# gdb -ex start --args ./a.out