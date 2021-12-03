mkdir bin
gcc -c program.S -o ./bin/program.o && ld ./bin/program.o -o ./bin/program
cat data.txt | ./bin/program | hexyl
# gdb -ex start --args ./a.out