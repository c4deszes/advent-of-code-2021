#pragma once

#define BOARD_SIZE 5

typedef struct {
	unsigned int field[BOARD_SIZE][BOARD_SIZE];
} board_t;

typedef struct {
	unsigned int field[BOARD_SIZE][BOARD_SIZE];
} board_state_t;

typedef struct {
	unsigned int win;
	unsigned int score;
} board_score_t;
