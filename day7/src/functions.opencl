/**
 * 
 */
__kernel void calc_fuel1(__global int* numbers, int numbers_size, __global int* fuel_buffer) {
	int index = get_global_id(0);
	int position = get_global_id(1);
	fuel_buffer[position * numbers_size + index] = abs(numbers[index] - position);
};

/**
 * 
 */
__kernel void calc_fuel2(__global int* numbers, int numbers_size, __global int* fuel_buffer) {
	int index = get_global_id(0);
	int position = get_global_id(1);
	fuel_buffer[position * numbers_size + index] = (abs(numbers[index] - position)+1) * abs(numbers[index] - position) / 2;
};

__kernel void sum_fuel(int numbers_size, __global int* fuel_buffer, __global int* sum_buffer) {
	int position = get_global_id(0);
	int total = 0;
	for (int i = 0; i < numbers_size; i++) {
		total += fuel_buffer[position * numbers_size + i];
	}
	sum_buffer[position] = total;
};

__kernel void min_fuel(int positions, __global int* sum_buffer, __global int* result_buffer) {
	int total = 0;
	int min_pos = 0;
	for (int i = 0; i < positions; i++) {
		if (sum_buffer[i] < sum_buffer[min_pos]) {
			min_pos = i;
		}
	}
	*result_buffer = sum_buffer[min_pos];
};
