# Advent of Code 2021

## Rules

+ Data cannot be manipulated with the following exceptions
  + Changing line and file endings is allowed e.g.: CRLF to LF, or adding newline with EOF character
  + "Platforming" the data is allowed as long as it's only used to communicate the data to the
    solution platform or to overcome it's Turing incompleteness. E.g.: converting numbers in text to
    binary '0' and '1' ASCII format is okay, converting numbers directly into binary directly is not

+ The output data must be complete, Intermediate solutions on which further manual calculations are
needed are not allowed. E.g.: binary output is allowed, but a binary output where you have to
manually calculate the amount of 1's is not allowed

---

## Day 1

Context: x86 Assembly

Registers: https://en.wikibooks.org/wiki/X86_Assembly/X86_Architecture
Instruction sheet: http://ref.x86asm.net/coder32.html
Instructions: https://www.felixcloutier.com/x86/mul

---

## Day 2

Context: CMake

---

## Day 3

Context: Bash

---

## Day 4

Context: CUDA (and C)

Bingo boards are first loaded onto GPU then the CPU instructs it to find matching numbers in parallel.
At each iteration vertical and horizontal matches are searched by the GPU and if there are any the
board will be marked and it's score will calculated.

The gameplay is CPU driven and the only logic where the CPU is more heavily involved is finding the
solution to the first and last wins.

---

## Day 5

Context: Minecraft

The solution I used spawned all the lines as blocks of sand in a minecraft world. The sand then fell
over each other and at each point where the lines would cross blocks could be counted.

Commands help:
Scheduling: https://minecraft.fandom.com/wiki/Commands/schedule
https://minecraft.fandom.com/wiki/Commands/fill

https://minecraft.fandom.com/wiki/Commands/execute#.28if.7Cunless.29_blocks

https://minecraft.fandom.com/wiki/Function_(Java_Edition)

---

## Day 6

Context: OpenCL and C++

### Solution 1

The cost of fuel for each initial position and new position is `abs(initial - new)`, finding the
minimum is simply adding the fuel cost for function for each crab like `abs(x-20) + abs(x-50) ..`
and then finding the global minimum of this function.

To solve this I used parallel processing to first calculate the fuel cost for each crab moving to
the Nth position, then summarized the cost again in parallel for each possible position and in the
last step a single thread goes through and finds the minimum position.
