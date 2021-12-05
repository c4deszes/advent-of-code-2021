#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {
	#include "solver.cuh"
	#include "model.h"
}

__global__ void update_state(board_t* board, board_state_t* state, unsigned int number) {
	int xi = threadIdx.x;
	int yi = threadIdx.y;
	if (board[blockIdx.x].field[yi][xi] == number) {
		state[blockIdx.x].field[yi][xi] = 1;
	}
}

__global__ void check_vertical(board_state_t* state, board_score_t* score) {
	int yi = threadIdx.x;
	for (int i = 0; i < BOARD_SIZE; i++) {
		if (state[blockIdx.x].field[yi][i] == 0) {
			return;
		}
	}
	score[blockIdx.x].win = 1;
}

__global__ void check_horizontal(board_state_t* state, board_score_t* score) {
	int xi = threadIdx.x;
	for (int i = 0; i < BOARD_SIZE; i++) {
		if (state[blockIdx.x].field[i][xi] == 0) {
			return;
		}
	}
	score[blockIdx.x].win = 1;
}

__global__ void calculate_score(board_t* board, board_state_t* state, board_score_t* score, unsigned int number) {
	if (score[blockIdx.x].win == 0 || score[blockIdx.x].score != 0) {
		return;
	}
	unsigned int local = 0;
	for (int y = 0; y < BOARD_SIZE; y++) {
		for (int x = 0; x < BOARD_SIZE; x++) {
			if (state[blockIdx.x].field[y][x] == 0) {
				local += board[blockIdx.x].field[y][x];
			}
		}
	}
	score[blockIdx.x].score = local * number;
}

static unsigned int board_count;
static board_t** boards;
static board_state_t** states;
static board_score_t* scores;

static void* boards_buffer;
static void* scores_buffer;
static void* states_buffer;

cudaError_t init_boards(board_t** l_boards, board_state_t** l_states, board_score_t* l_scores, unsigned int l_board_count) {
	int nDevices = 0;
	cudaGetDeviceCount(&nDevices);
	if (!nDevices) {
		printf("No CUDA capable GPU was found.\n");
		return cudaErrorDevicesUnavailable;
	}

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("Device name: %s\n", prop.name);

	boards = l_boards;
	states = l_states;
	scores = l_scores;
	board_count = l_board_count;

	//
	cudaMalloc(&boards_buffer, board_count * sizeof(board_t));
	for (int i = 0; i < board_count; i++) {
		cudaMemcpy((void*)((size_t)boards_buffer + i * sizeof(board_t)), boards[i], sizeof(board_t), cudaMemcpyHostToDevice);
	}
	printf("Copied boards.\n");

	//
	cudaMalloc(&scores_buffer, board_count * sizeof(board_score_t));
	cudaMemcpy(scores_buffer, scores, board_count * sizeof(board_score_t), cudaMemcpyHostToDevice);
	printf("Copied board scores.\n");

	//
	cudaMalloc(&states_buffer, board_count * sizeof(board_state_t));
	for (int i = 0; i < board_count; i++) {
		cudaMemcpy((void*)((size_t)states_buffer + i * sizeof(board_state_t)), states[i], sizeof(board_state_t), cudaMemcpyHostToDevice);
	}
	printf("Copied states.\n");

	return cudaSuccess;
}

cudaError_t update_boards(unsigned int number) {
	//printf("Playing: %d\n", number);
	dim3 blocks(board_count);
	dim3 threads_full(5, 5);
	dim3 threads_stripe(5);
	dim3 threads_single(1);
	update_state<<< blocks, threads_full >>>((board_t*)boards_buffer, (board_state_t*) states_buffer, number);
	check_vertical<<< blocks, threads_stripe >>>((board_state_t*) states_buffer, (board_score_t*) scores_buffer);
	check_horizontal<<< blocks, threads_stripe >>>((board_state_t*) states_buffer, (board_score_t*) scores_buffer);
	calculate_score<<< blocks, threads_single >>>((board_t*)boards_buffer, (board_state_t*) states_buffer, (board_score_t*) scores_buffer, number);

	cudaError_t status = cudaDeviceSynchronize();
	if (status != cudaSuccess) {
		printf("Error %d\n", status);
	}

	status = cudaMemcpy(scores, scores_buffer, board_count * sizeof(board_score_t), cudaMemcpyDeviceToHost);
	if (status != cudaSuccess) {
		printf("Error copying scores buffer: %d", status);
	}

	// Copy back result
	for (int i = 0; i < board_count; i++) {
		status = cudaMemcpy(states[i], (void*)((size_t)states_buffer + i * sizeof(board_state_t)), sizeof(board_state_t), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess) {
			printf("Error copying statebuffer: %d", status);
		}
	}

	// printf("Last Error: %d\n", cudaGetLastError());
	return status;
}

cudaError_t destroy_boards() {
	cudaError_t status;

	status = cudaFree(boards_buffer);
	if (status != cudaSuccess) {
		printf("Error freeing board buffer: %d", status);
	}
	status = cudaFree(scores_buffer);
	if (status != cudaSuccess) {
		printf("Error freeing scores buffer: %d", status);
	}
	status = cudaFree(states_buffer);
	if (status != cudaSuccess) {
		printf("Error freeing board buffer: %d", status);
	}

	return cudaSuccess;
}
