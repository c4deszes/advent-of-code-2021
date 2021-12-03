# -------------------------------------------------------------
#
# Using Linux System calls for 64-bit
# to run:
#   gcc -c hello.s && ld hello.o && ./a.out
# o
#   gcc -nostdlib hello.s && ./a.out
#
# --------------------------------------------------------------

.global _start

.text

# Entry point for the program
_start:

_loop:
    call _read_line
    
    # If EOF then exit
    mov $eof_flag, %rsi
    mov (%rsi), %al
    cmp $1, %al
    je _exit
    
    call _parse
    call _convert
    call _print

    # Load last and previous value
    mov $last_value, %rsi
    mov (%rsi), %eax
    mov $prev_value, %rsi
    mov (%rsi), %ecx
    mov %eax, prev_value
    cmp %ecx, %eax          # Compare values
    jbe _loop              # If decreased keep going
    
    mov $result, %rsi       # Else increase result counter
    mov (%rsi), %ecx
    inc %ecx
    mov %ecx, (%rsi)
    jmp _loop               # Keep going

_exit:
    mov $result, %rsi       # Else increase result counter
    mov (%rsi), %eax
    dec %eax
    mov %eax, stdin_buffer
    call _print

    # exit(0)
    mov $60, %rax           # system call 60 is exit
    mov $0, %rdi
    syscall

_print:
    # Print Result
    mov $1, %rax            # system call 1 is write
    mov $1, %rdi            # file handler 1 is stdout
    mov $stdin_buffer, %rsi      # address of string to output
    mov $8, %rdx           # number of bytes
    syscall
    ret

# Read until a newline character
_read_line:
    mov $stdin_buffer, %rsi
_read_line_loop:
    # Read 1 byte from STDIN
    mov $0, %rax
    mov $0, %rdi
    mov $1, %rdx
    syscall

    # Compare against new line character
    mov (%rsi), %dil	# Load last read value
    cmp $0, %dil
    je _read_line_eof  
    cmp $10, %dil		# Compare to newline
    je _read_line_exit

    # Increment and repeat
    inc %rsi
    jmp _read_line_loop
_read_line_eof:
    mov $1, %al
    mov $eof_flag, %rsi
    mov %al, (%rsi)
_read_line_exit:
    ret

_parse:
    mov $stdin_buffer, %rsi
_parse_loop:
    mov (%rsi), %dil    # Load last read value
    cmp $10, %dil       # Compare to newline
    je _parse_exit
    
    # Offset character so we get it's numerical value
    sub $48, %dil		# 48 is Decimal for the ASCII character '0'
    mov %dil, (%rsi)    # Replace existing value
    inc %rsi            # Increment array index
    jmp _parse_loop
_parse_exit:
    ret

_convert:
    mov $stdin_buffer, %rsi
_convert_forward:
    mov (%rsi), %dil        # Load last read value
    inc %rsi                # Increment array index
    cmp $10, %dil           # Compare to newline
    jne _convert_forward    
    dec %rsi                
    dec %rsi                

    mov $1, %bx
    mov %bx, temp
    mov $0, %di            # Initialize accumulate RDX = 0
_convert_loop:
    mov (%rsi), %al    # Load Data DIL = *(RSI)
    mov temp, %bx
    mul %bx
    # movzx %al, %rdi
    add %ax, %di
    # mov %ebx, %edi
    dec %rsi            # Decrement array index
    
    mov temp, %ax
    mov $10, %bx
    mul %bx             # Multiply EAX = EAX * 10
    mov %ax, temp       # Store multiplier
    
    mov $stdin_buffer, %rbx
    dec %rbx
    cmp %rbx, %rsi
    jne _convert_loop
    
    mov %edi, stdin_buffer
    mov %edi, last_value
    ret

.data
stdin_buffer:
    .space 32
eof_flag:
    .space 1
temp:
    .space 4
result:
    .space 4
prev_value:
    .space 4
last_value:
    .space 4
