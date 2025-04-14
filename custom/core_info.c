static int _kt_thread_id_to_core[] = {
/* 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 */
   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20,
/* 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 */
   24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47
};

static int _kt_prefill_core_enabled[] = {
/* 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 */
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
/* 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 */
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

static int _kt_decode_core_enabled[] = {
/* 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 */
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
/* 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 */
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
};

long kt_worker_thread_idle_threshold() {
    return 410000000; // cpu Hz / 10
}

int kt_thread_id_to_numa(int thread_id) {
    if (0 <= thread_id && thread_id <= 20)
        return 0;
    else
        return 1;
}

int kt_thread_id_to_core(int thread_id) {
    return _kt_thread_id_to_core[thread_id];
}

int kt_prefill_core_enabled(int thread_id) {
    return _kt_prefill_core_enabled[thread_id];
}

int kt_decode_core_enabled(int thread_id) {
    return _kt_decode_core_enabled[thread_id];
}

int kt_thread_id_to_steal_from(int thread_id) {
    if (0 <= thread_id && thread_id <= 20)
        return 0;
    else
        return 21;
}

int kt_thread_id_to_steal_to(int thread_id) {
    if (0 <= thread_id && thread_id <= 20)
        return 21;
    else
        return 45;
}
