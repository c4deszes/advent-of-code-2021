#!/bin/bash
ones=(0)	#7 5 8 7 5
zeros=(0)	#5 7 4 5 7
lines=()
while IFS= read -r LINE
do
	i=0
	lines+=($LINE)
	LINE=$(echo $LINE | fold -w 1)
	for c in ${LINE}; do
		if [ ${#ones[@]} -lt $[$i+1] ]; then
			ones+=(0)
			zeros+=(0)
		fi

		if [[ $c == *"1"* ]]; then
			ones[$i]=$[${ones[$i]}+1]
		elif [[ $c == *"0"* ]]; then
			zeros[$i]=$[${zeros[$i]}+1]
		else
			echo $c
		fi
		i=$[i+1]
	done
done < /dev/stdin

for line in ${lines}; do
	for ((i=0;i<LENGTH;i++)); do
		if [ ${zeros[$i]} -gt ${ones[$i]} ]; then
			GAMMA+='0'
			EPSILON+='1'
		else
			GAMMA+='1'
			EPSILON+='0'
		fi
	done
done