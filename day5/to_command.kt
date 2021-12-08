package com.app         // 1

import java.io.File
import kotlin.collections.List

fun lineCommand(x1: Int, y1: Int, x2: Int, y2: Int, z: Int): String {
	return "fill ${x1} ${z} ${y1} ${x2} ${z} ${y2} sand";
}

fun main() {
	var lines = File("data.txt").bufferedReader().readLines();
	var pattern = "([0-9]+),([0-9]+) -> ([0-9]+),([0-9]+)".toRegex()
	var fileCounter = 0
	var z = 60
	var loadFile = File("load${fileCounter}.mcfunction").bufferedWriter()
	for (line in lines) {
		var match = pattern.matchEntire(line)
		if (match != null) {
			var x1 = match.groupValues.get(1).toInt()
			var y1 = match.groupValues.get(2).toInt()
			var x2 = match.groupValues.get(3).toInt()
			var y2 = match.groupValues.get(4).toInt()

			// Reset Z coordinate
			if (z > 200) {
				z = 60
				fileCounter++
				loadFile.flush()
				loadFile.close()
				loadFile = File("load${fileCounter}.mcfunction").bufferedWriter()
			}

			if (x1 != x2 && y1 != y2) {
				continue
			}

			loadFile.write(lineCommand(x1, y1, x2, y2, z))
			loadFile.write("\n")
			z++
		}
	}
	loadFile.flush()
	loadFile.close()

	// Generating common load script
	loadFile = File("load.mcfunction").bufferedWriter()
	loadFile.write("scoreboard objectives add temp dummy\n");
	loadFile.write("scoreboard objectives add count dummy\n");
	loadFile.write("scoreboard objectives setdisplay sidebar count\n");
	for (i in 0..fileCounter) {
		loadFile.write("schedule function challenge:load${fileCounter} 15s append\n")
	}
	loadFile.flush()
	loadFile.close()

	// Generating calculating script
	loadFile = File("calculate.mcfunction").bufferedWriter()
	loadFile.write("scoreboard players set c4deszes count 0\n")
	var y = 0
	while (y < 1100) {
		var x = 0
		while (x < 1100) {
			loadFile.write("execute store result score c4deszes temp if blocks ${x} ${z} ${y} ${x+180} ${z} ${y+180} ${x} ${z} ${y} masked\n")
			loadFile.write("scoreboard players operation c4deszes count += c4deszes temp\n")
			x += 181
		}
		y+=181
	}
	loadFile.flush()
	loadFile.close()
}
