//
// Created by rakshitgl
//

kernel void parallel_adder(global float* a, global float* b, global float* results) {
    int i = get_global_id(0);
    results[i] = a[i] + b[i];
}

kernel void parallel_subtracter(global float* a, global float* b, global float* results) {
    int i = get_global_id(0);
    results[i] = a[i] - b[i];
}

kernel void parallel_multiplier(global float* a, global float* b, global float* results) {
    int i = get_global_id(0);
    results[i] = a[i] * b[i];
}

kernel void parallel_gt(global float* a, global float* b, global float* results) {
    int i = get_global_id(0);
    results[i] = a[i] > b[i];
}

kernel void parallel_lt(global float* a, global float* b, global float* results) {
    int i = get_global_id(0);
    results[i] = a[i] < b[i];
}

kernel void parallel_equals(global float* a, global float* b, global float* results) {
    int i = get_global_id(0);
    results[i] = a[i] == b[i];
}

kernel void parallel_gte(global float* a, global float* b, global float* results) {
    int i = get_global_id(0);
    results[i] = a[i] >= b[i];
}

kernel void parallel_lte(global float* a, global float* b, global float* results) {
    int i = get_global_id(0);
    results[i] = a[i] <= b[i];
}

kernel void scalar_parallel_multiplier(global float* a, global float* b, global float* results) {
    int i = get_global_id(0);
    results[i] = a[i] * b[0];
}

kernel void scalar_parallel_gt(global float* a, global float* b, global float* results) {
    int i = get_global_id(0);
    results[i] = a[i] > b[0];
}

kernel void scalar_parallel_lt(global float* a, global float* b, global float* results) {
    int i = get_global_id(0);
    results[i] = a[i] < b[0];
}

kernel void scalar_parallel_equals(global float* a, global float* b, global float* results) {
    int i = get_global_id(0);
    results[i] = a[i] == b[0];
}

kernel void scalar_parallel_gte(global float* a, global float* b, global float* results) {
    int i = get_global_id(0);
    results[i] = a[i] >= b[0];
}

kernel void scalar_parallel_lte(global float* a, global float* b, global float* results) {
    int i = get_global_id(0);
    results[i] = a[i] <= b[0];
}

kernel void scalar_parallel_power(global float* a, global float* b, global float* results) {
    int i = get_global_id(0);
    results[i] = pow(a[i],b[0]);
}

kernel void scalar_parallel_adder(global float* a, global float* b, global float* col_size, global float* results) {
    int i = get_global_id(0);

    int r = i/(int)col_size[0];
    int c = i%(int)col_size[0];

    results[i] = a[i];

    if(r==c)
        results[i] = results[i] + b[0];
}

kernel void scalar_parallel_subtracter(global float* a, global float* b, global float* col_size, global float* results) {
    int i = get_global_id(0);

    int r = i/(int)col_size[0];
    int c = i%(int)col_size[0];

    results[i] = a[i];

    if(r==c)
        results[i] = results[i] - b[0];
}

kernel void parallel_matrix_multiply(const int M, const int N, const int K, const global float* A, const global float* B, global float* C) {
    
    const int row = get_global_id(0);
    const int col = get_global_id(1);

    float sum = 0.0f;
    for (int k=0; k<K; k++) {
        sum += A[k*M + row] * B[col*K + k];
    }

    C[col*M + row] = sum;
}
