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

// void print() {
// 	for (int i = 0; i < 2; i++) {
// 		printf("Board%d:\n", i);
// 		for (int y = 0; y < 5; y++) {
// 			for (int x = 0; x < 5; x++) {
// 				printf("%d\t", states[i]->field[y][x]);
// 			}
// 			printf("\n");
// 		}
// 		printf("\n");
// 	}
// }

void print_scores() {
	for (int i = 0; i < board_count; i++) {
		printf("Board %d win:%d\n", i, scores[i].score);
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

	// Debug number series
	// for (int i = 0; i < number_index; i++) {
	// 	printf("%d\n", number_series[i]);
	// }

	// Read dummy newline
	scanf("\n");

	unsigned int field_index = 0;
	unsigned int field[5][5];

	while ((res = scanf("%d %d %d %d %d", &(field[field_index][0]), &(field[field_index][1]), &(field[field_index][2]), &(field[field_index][3]), &(field[field_index][4]))) != EOF) {
		field_index++;

		if (field_index == 5) {
			// copy
			field_index = 0;

			// New Board
			void* field_ptr = malloc(sizeof(field));
			memcpy(field_ptr, field, sizeof(field));
			boards[board_count] = field_ptr;

			void* state_ptr = malloc(sizeof(board_state_t));
			memcpy(state_ptr, &default_state, sizeof(board_state_t));
			states[board_count] = state_ptr;

			board_count++;

			// Debug boards
			// for (int y = 0; y < 5; y++) {
			// 	for (int x = 0; x < 5; x++) {
			// 		printf("%d - ", field[y][x]);
			// 	}
			// 	printf("\n");
			// }
			// printf("\n");
		}

		char terminal = '\0';
		scanf("%c", &terminal);
		if (terminal == EOF) {
			break;
		}
	}

	init_boards(boards, states, scores, board_count);

	for (int i = 0; i< number_index; i++) {
		update_boards(number_series[i]);

		for (int k = 0; k < board_count; k++) {
			if (scores[k].win == 1) {
				goto exit;
			}
		}
	}
	exit:
	// update_boards(4);

	destroy_boards();

	// print();

	print_scores();

	return 0;
}