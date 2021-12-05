#pragma once

#include "model.h"
#include <cuda.h>
#include <cuda_runtime.h>

cudaError_t init_boards(board_t** l_boards, board_state_t** l_states, board_score_t* l_scores, unsigned int l_board_count);

cudaError_t update_boards(unsigned int number);

cudaError_t destroy_boards();
