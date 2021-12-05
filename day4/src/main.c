#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "solver.cuh"

board_state_t default_state = {
	.field = {
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0},
		{0, 0, 0, 0, 0}
	}
};

unsigned int board_count = 0;
board_t* boards[200];
board_state_t* states[200];
board_score_t scores[200];

void printBoard(int i) {
	printf("Board%d:\n", i);
	printf("Numbers:\n");
	for (int y = 0; y < 5; y++) {
		for (int x = 0; x < 5; x++) {
			printf("%d\t", boards[i]->field[y][x]);
		}
		printf("\n");
	}

	printf("State:\n", i);
	for (int y = 0; y < 5; y++) {
		for (int x = 0; x < 5; x++) {
			printf("%d\t", states[i]->field[y][x]);
		}
		printf("\n");
	}
	printf("\n");
}

void print_scores() {
	for (int i = 0; i < board_count; i++) {
		printf("Board %d win=%d, score=%d\n", i, scores[i].win, scores[i].score);
	}
}

unsigned int number_series[500];
unsigned int number_index = 0;

int main() {

	int res;
	unsigned int number = 0;

	while ((res = scanf("%d", &number)) != EOF) {

		number_series[number_index] = number;
		number_index++;

		char terminal = '\0';
		scanf("%c", &terminal);
		if (terminal == ',') {
			continue;
		}
		else if (terminal == '\n' || terminal == EOF) {
			break;
		}
	}

	// Read dummy newline
	scanf("\n");

	unsigned int field_index = 0;
	unsigned int field[5][5];

	while ((res = scanf("%d %d %d %d %d", &(field[field_index][0]), &(field[field_index][1]), &(field[field_index][2]), &(field[field_index][3]), &(field[field_index][4]))) != EOF) {
		field_index++;

		if (field_index == 5) {
			void* field_ptr = malloc(sizeof(field));
			memcpy(field_ptr, field, sizeof(field));
			boards[board_count] = field_ptr;

			void* state_ptr = malloc(sizeof(board_state_t));
			memcpy(state_ptr, &default_state, sizeof(board_state_t));
			states[board_count] = state_ptr;

			field_index = 0;
			board_count++;
		}

		char terminal = '\0';
		scanf("%c", &terminal);
		if (terminal == EOF) {
			break;
		}
	}

	init_boards(boards, states, scores, board_count);

	int i = 0;
	int winning_board_index = 0;
	while(i< number_index) {
		update_boards(number_series[i]);
		//print_scores();

		for (int k = 0; k < board_count; k++) {
			if (scores[k].win == 1) {
				printf("Solution 1: %d\n", scores[k].score);
				winning_board_index = k;
				goto exit;
			}
		}
		i++;
	}
	exit:
	int boards_left = 0;
	int last_index = 0;

	while(i < number_index) {
		boards_left = 0;
		last_index = 0;

		for (int k = 0; k < board_count; k++) {
			if (scores[k].win == 0) {
				last_index = k;
				boards_left++;
			}
		}
		if (boards_left == 1) {
			break;
		}
		update_boards(number_series[i]);
		//print_scores();
		i++;
	}
	while(scores[last_index].win != 1) {
	   	update_boards(number_series[i]);
	   	i++;
	}
	printf("Solution 2: %d, index=%d\n", scores[last_index].score, last_index);

	destroy_boards();
	//print_scores();
	//printBoard(winning_board_index);

	//printBoard(0);
	//printBoard(1);
	//printBoard(2);

	return 0;
}