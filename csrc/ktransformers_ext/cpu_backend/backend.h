/**
 * @Description  :
 * @Author       : chenht2022
 * @Date         : 2024-07-22 02:03:05
 * @Version      : 1.0.0
 * @LastEditors  : chenht2022
 * @LastEditTime : 2024-07-25 10:33:38
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_BACKEND_H
#define CPUINFER_BACKEND_H

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

enum ThreadStatus {
    WORKING,
    WAITING,
    EXIT,
};

#define use_yield 0
#define use_shlock 0
#if use_shlock
    #define KT_LOCK "/dev/shm/kt.lock"
#endif
#define use_epoll 0
#define use_poll 0
#define numa_atomic 1
#if numa_atomic && defined(USE_NUMA)
    #define CACHE_LINE (64)
    #define CPU_CORE_COUNT (1024)

struct kt_atomic {
    std::atomic<int>          curr   __attribute__((aligned(CACHE_LINE)));
    std::atomic<ThreadStatus> status __attribute__((aligned(CACHE_LINE)));

    kt_atomic(): curr(0), status(ThreadStatus::WAITING) {}
};

struct kt_atomic_numa {
    struct kt_atomic atomics[CPU_CORE_COUNT];
    char padding[1024 * 1024 * 16];
};
// 超大对象会被放在mimalloc的2M巨页中，方便观察numa分配

struct ThreadState {
    std::shared_ptr<struct kt_atomic_numa> numa0 __attribute__((aligned(CACHE_LINE)));
    std::shared_ptr<struct kt_atomic_numa> numa1 __attribute__((aligned(CACHE_LINE)));

    char padding[CACHE_LINE];

    std::atomic<ThreadStatus>* status;
    std::atomic<int>*          curr;
    int end;
};
#else
struct ThreadState {
    std::unique_ptr<std::atomic<ThreadStatus>> status;
    std::unique_ptr<std::atomic<int>> curr;
    int end;
};
#endif

class Backend {
  public:
    Backend(int);
    ~Backend();
    int get_thread_num();
    void do_work_stealing_job(int, std::function<void(int)>,
                              std::function<void(int)>,
                              std::function<void(int)>);
    static thread_local int numa_node;
    static thread_local int steal_from;
    static thread_local int steal_to;
    static thread_local int thread_local_id;

  private:
#if use_shlock
    static std::atomic_bool* shlock;
#endif
#if use_epoll || use_poll
    int evfd[1024]; // per thread
    #if use_epoll
    int epfd[1024];
    #endif
#endif
    int thread_num_;
    int max_thread_num_;
    std::vector<ThreadState> thread_state_; // [thread_num]
    std::function<void(int)> init_func_;
    std::function<void(int)> compute_func_;
    std::function<void(int)> finalize_func_;
    std::vector<std::thread> workers_;

    void process_tasks(int);
    void worker_thread(int);
};
#endif
