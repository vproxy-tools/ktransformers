/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:05
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:33:34
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif // _GNU_SOURCE

#include <unistd.h>
#include <sched.h>

#ifndef likely
#define likely(x)       __builtin_expect(!!(x), 1)
#endif // likely
#ifndef unlikely
#define unlikely(x)     __builtin_expect(!!(x), 0)
#endif // unlikely

#include "backend.h"
#include "core_info.h"
#include <pthread.h>

#ifdef USE_NUMA
#include <numa.h>
#include <numaif.h>

thread_local int Backend::numa_node = -1;
thread_local int Backend::steal_from = -1;
thread_local int Backend::steal_to = -1;
#endif

thread_local int Backend::thread_local_id = -1;

Backend::Backend(int max_thread_num) {
    max_thread_num_ = max_thread_num;
    thread_state_.resize(max_thread_num_);
    for (int i = 0; i < max_thread_num_; i++) {
        thread_state_[i].curr = std::make_unique<std::atomic<int>>();
        thread_state_[i].status =
            std::make_unique<std::atomic<ThreadStatus>>(ThreadStatus::WAITING);
    }
    workers_.resize(max_thread_num_);
    for (int i = 1; i < max_thread_num_; i++) {
        workers_[i] = std::thread(&Backend::worker_thread, this, i);
    }
}

Backend::~Backend() {
    for (int i = 0; i < max_thread_num_; i++) {
        thread_state_[i].status->store(ThreadStatus::EXIT,
                                       std::memory_order_release);
    }
    for (int i = 1; i < max_thread_num_; i++) {
        if (workers_[i].joinable()) {
            workers_[i].join();
        }
    }
}

int Backend::get_thread_num() { return max_thread_num_; }

void Backend::do_work_stealing_job(int task_num,
                                   std::function<void(int)> init_func,
                                   std::function<void(int)> compute_func,
                                   std::function<void(int)> finalize_func) {
    init_func_ = init_func;
    compute_func_ = compute_func;
    finalize_func_ = finalize_func;
#ifdef USE_NUMA
    // numa node location will be calculated based on the number of threads
    thread_num_ = max_thread_num_;
#else
    thread_num_ = std::min(max_thread_num_, task_num);
#endif
    int base = task_num / thread_num_;
    int remain = task_num % thread_num_;
    thread_state_[0].end = base + (0 < remain);

    // 为主线程设置 thread_local_id
    thread_local_id = 0;

    for (int i = 1; i < thread_num_; i++) {
        thread_state_[i].curr->store(thread_state_[i - 1].end,
                                     std::memory_order_relaxed);
        thread_state_[i].end = thread_state_[i - 1].end + base + (i < remain);
        thread_state_[i].status->store(ThreadStatus::WORKING,
                                       std::memory_order_release);
    }
    thread_state_[0].curr->store(0, std::memory_order_relaxed);
    thread_state_[0].status->store(ThreadStatus::WORKING,
                                   std::memory_order_release);
    process_tasks(0);
    for (int i = 1; i < thread_num_; i++) {
        while (thread_state_[i].status->load(std::memory_order_acquire) ==
               ThreadStatus::WORKING) {
        }
    }
}

void Backend::process_tasks(int thread_id) {
    
    #ifdef USE_NUMA
    if(unlikely(numa_node == -1)){
        char thread_name[36];
        sprintf(thread_name, "llama.cpp:%d", thread_id);
        pthread_setname_np(pthread_self(), thread_name);

        numa_node = thread_id_to_numa[thread_id];

        struct bitmask* mask = numa_bitmask_alloc(numa_num_configured_nodes());
        numa_bitmask_setbit(mask, numa_node);
        numa_bind(mask);

        steal_from = thread_id_to_steal_from[thread_id];
        steal_to = thread_id_to_steal_to[thread_id];

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);

        int cpuid = thread_id_to_core[thread_id];

        CPU_SET(cpuid, &cpuset);
        sched_setaffinity(gettid(), sizeof(cpuset), &cpuset);
printf("thread_id = %d, nodes = %d, thread_num = %d, numa_node = %d, cpuid = %d, steal_from = %d, steal_to = %d\n", thread_id, numa_num_configured_nodes(), thread_num_, numa_node, cpuid, steal_from, steal_to);
fflush(stdout);
    }
    #endif

    if (init_func_ != nullptr) {
        init_func_(thread_id);
    }
    while (true) {
        int task_id = thread_state_[thread_id].curr->fetch_add(
            1, std::memory_order_acq_rel);
        if (task_id >= thread_state_[thread_id].end) {
            break;
        }
        compute_func_(task_id);
    }
    int steal_total = steal_to - steal_from;
    for (int t_offset = 1; t_offset < steal_total; t_offset++) {
        int t_i = (thread_id - steal_from + t_offset) % steal_total + steal_from;
        if (thread_state_[t_i].status->load(std::memory_order_acquire) !=
            ThreadStatus::WORKING) {
            continue;
        }
        while (true) {
            int task_id = thread_state_[t_i].curr->fetch_add(
                1, std::memory_order_acq_rel);
            if (task_id >= thread_state_[t_i].end) {
                break;
            }
            compute_func_(task_id);
        }
    }
    if (finalize_func_ != nullptr) {
        finalize_func_(thread_id);
    }
    thread_state_[thread_id].status->store(ThreadStatus::WAITING,
                                           std::memory_order_release);
}

void Backend::worker_thread(int thread_id) {
    uint64_t idle = 0;
    thread_local_id = thread_id; // 设置线程本地变量
    while (true) {
        ThreadStatus status =
            thread_state_[thread_id].status->load(std::memory_order_acquire);
        if (likely(status == ThreadStatus::WORKING)) {
            process_tasks(thread_id);
            idle = 0;
        } else if (status == ThreadStatus::WAITING) {
            if (++idle > worker_thread_idle_threshold) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } else if (status == ThreadStatus::EXIT) {
            return;
        }
    }
}
