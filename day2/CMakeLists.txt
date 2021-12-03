cmake_minimum_required(VERSION 3.16)
project(day2)

set(INPUT_FILE "data.txt")

# Load data file
file(READ ${INPUT_FILE} LINES)
string(REPLACE [[;]] [[\;]] LINES "${LINES}")
string(REPLACE "\n" ";" LINES "${LINES}")

# First challenge
set(HORIZONTAL_POS "0")
set(VERTICAL_POS "0")

foreach(LINE ${LINES})
	string(REGEX MATCH "((forward)|(up)|(down)) ([0-9])" MATCH_VAR ${LINE})
	set(LINE_COMMAND ${CMAKE_MATCH_1})
	set(LINE_PARAM ${CMAKE_MATCH_5})

	if (MATCH_VAR)
		if ("${LINE_COMMAND}" STREQUAL "forward")
			math(EXPR HORIZONTAL_POS "${HORIZONTAL_POS} + ${LINE_PARAM}" OUTPUT_FORMAT DECIMAL)
		elseif("${LINE_COMMAND}" STREQUAL "up")
			math(EXPR VERTICAL_POS "${VERTICAL_POS} - ${LINE_PARAM}" OUTPUT_FORMAT DECIMAL)
		elseif("${LINE_COMMAND}" STREQUAL "down")
			math(EXPR VERTICAL_POS "${VERTICAL_POS} + ${LINE_PARAM}" OUTPUT_FORMAT DECIMAL)
		endif()
	endif()
endforeach()

math(EXPR SOLUTION "${HORIZONTAL_POS} * ${VERTICAL_POS}" OUTPUT_FORMAT DECIMAL)
message("First Solution = ${SOLUTION}")

# Second challenge
set(HORIZONTAL_POS "0")
set(VERTICAL_POS "0")
set(AIM "0")

foreach(LINE ${LINES})
	string(REGEX MATCH "((forward)|(up)|(down)) ([0-9])" MATCH_VAR ${LINE})
	set(LINE_COMMAND ${CMAKE_MATCH_1})
	set(LINE_PARAM ${CMAKE_MATCH_5})

	if (MATCH_VAR)
		if ("${LINE_COMMAND}" STREQUAL "forward")
			math(EXPR HORIZONTAL_POS "${HORIZONTAL_POS} + ${LINE_PARAM}" OUTPUT_FORMAT DECIMAL)
			math(EXPR VERTICAL_POS "${VERTICAL_POS} + ${LINE_PARAM} * ${AIM}" OUTPUT_FORMAT DECIMAL)
		elseif("${LINE_COMMAND}" STREQUAL "up")
			math(EXPR AIM "${AIM} - ${LINE_PARAM}" OUTPUT_FORMAT DECIMAL)
		elseif("${LINE_COMMAND}" STREQUAL "down")
			math(EXPR AIM "${AIM} + ${LINE_PARAM}" OUTPUT_FORMAT DECIMAL)
		endif()
	endif()
endforeach()

math(EXPR SOLUTION "${HORIZONTAL_POS} * ${VERTICAL_POS}" OUTPUT_FORMAT DECIMAL)
message("Second Solution = ${SOLUTION}")