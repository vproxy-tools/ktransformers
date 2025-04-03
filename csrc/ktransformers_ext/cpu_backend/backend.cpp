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
#if use_shlock
    #include <sys/shm.h>
#endif
#include <sys/mman.h>
#include <fcntl.h>

#ifndef likely
#define likely(x)       __builtin_expect(!!(x), 1)
#endif // likely
#ifndef unlikely
#define unlikely(x)     __builtin_expect(!!(x), 0)
#endif // unlikely

#include "backend.h"
#include "core_info.h"
#include <pthread.h>

#include <numa.h>
#include <numaif.h>

#if use_epoll || use_poll
    #if use_poll
        #include <poll.h>
    #else
        #include <sys/epoll.h>
    #endif
    #include <sys/eventfd.h>
#endif

thread_local int Backend::numa_node = -1;
thread_local int Backend::steal_from = -1;
thread_local int Backend::steal_to = -1;

thread_local int Backend::thread_local_id = -1;
#if use_shlock
std::atomic_bool* Backend::shlock = nullptr;
#endif

Backend::Backend(int max_thread_num) {
    max_thread_num_ = max_thread_num;

#if use_epoll || use_poll
    for (int i = 1; i < max_thread_num; ++i) {
        evfd[i] = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
        #if use_epoll
        epfd[i] = epoll_create(1);
        struct epoll_event evt = {
            .events = EPOLLIN | EPOLLET,
        };
        epoll_ctl(epfd[i], EPOLL_CTL_ADD, evfd[i], &evt);
        #endif
    }
#endif

#if numa_atomic && defined(USE_NUMA)
    int oldpolicy;
    struct bitmask* oldmask = numa_allocate_nodemask();
    if (get_mempolicy(&oldpolicy, oldmask->maskp,
                      oldmask->size + 1, 0, 0) < 0) {
printf("get_mempolicy failed, errno=%d %s\n", errno, strerror(errno));
fflush(stdout);
        oldpolicy = MPOL_DEFAULT;
    }
#endif

    thread_state_.resize(max_thread_num_);
    workers_.resize(max_thread_num_);

#if numa_atomic && defined(USE_NUMA)
    numa_set_preferred(0);
    auto numa0 = std::make_shared<struct kt_atomic_numa>();
    numa_set_preferred(1);
    auto numa1 = std::make_shared<struct kt_atomic_numa>();
#endif

    thread_state_.resize(max_thread_num_);
    for (int i = 0; i < max_thread_num_; i++) {
#if numa_atomic && defined(USE_NUMA)
        struct kt_atomic* atomic;
        if (thread_id_to_numa[i] == 0) {
            atomic = &numa0->atomics[i];
        } else {
            atomic = &numa1->atomics[i];
        }

        thread_state_[i].curr = &atomic->curr;
        thread_state_[i].status = &atomic->status;

        thread_state_[i].numa0 = numa0;
        thread_state_[i].numa1 = numa1;
#else
        thread_state_[i].curr = std::make_unique<std::atomic<int>>();
        thread_state_[i].status =
            std::make_unique<std::atomic<ThreadStatus>>(ThreadStatus::WAITING);
#endif
    }
    for (int i = 1; i < max_thread_num_; i++) {
#if numa_atomic && defined(USE_NUMA)
        numa_set_preferred(thread_id_to_numa[i]);
#endif
        workers_[i] = std::thread(&Backend::worker_thread, this, i);
    }

#if numa_atomic && defined(USE_NUMA)
    if (oldpolicy == MPOL_DEFAULT) {
        numa_set_localalloc();
    } else {
        set_mempolicy(oldpolicy, oldmask->maskp,
                      oldmask->size + 1);
    }
    numa_free_cpumask(oldmask);
#endif
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
#if use_shlock
    if (unlikely(shlock == nullptr)) {
        bool is_new = access(KT_LOCK, F_OK) != 0;
        int shmfd = open(KT_LOCK, O_CREAT | O_RDWR, 0600);
        size_t mmsize = sizeof(std::atomic_bool);
        if (is_new) {
            ftruncate(shmfd, mmsize);
        }
        shlock = (std::atomic_bool*) mmap(NULL, mmsize, PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_POPULATE,
                shmfd, 0);
printf("fd = %d, lock = %p, last errno = %d %s\n", shmfd, shlock, errno, strerror(errno));
fflush(stdout);
        if (is_new) {
            memset(shlock, 0, mmsize);
        }
        close(shmfd);
    }
#endif

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

#if use_shlock
    bool bfalse = false;
    while (!shlock->compare_exchange_weak(bfalse, true, std::memory_order_acquire, std::memory_order_relaxed)) {
        bfalse = false;
    }
#endif

    for (int i = 1; i < thread_num_; i++) {
        thread_state_[i].curr->store(thread_state_[i - 1].end,
                                     std::memory_order_relaxed);
        thread_state_[i].end = thread_state_[i - 1].end + base + (i < remain);
        thread_state_[i].status->store(ThreadStatus::WORKING,
                                       std::memory_order_release);
    }

#if use_epoll || use_poll
    uint64_t dummy = 1;
    for (int i = 1; i < thread_num_; i++) {
        write(evfd[i], &dummy, 8);
    }
#endif

    thread_state_[0].curr->store(0, std::memory_order_relaxed);
    thread_state_[0].status->store(ThreadStatus::WORKING,
                                   std::memory_order_release);
    process_tasks(0);
    for (int i = 1; i < thread_num_; i++) {
        while (thread_state_[i].status->load(std::memory_order_acquire) ==
               ThreadStatus::WORKING) {
        }
    }

#if use_shlock
    shlock->store(false, std::memory_order_release);
#endif
}

void Backend::process_tasks(int thread_id) {
    
    if(unlikely(numa_node == -1)){
        char thread_name[36];
        sprintf(thread_name, "llama.cpp:%d", thread_id);
        pthread_setname_np(pthread_self(), thread_name);

#ifdef USE_NUMA
        numa_node = thread_id_to_numa[thread_id];
        struct bitmask* mask = numa_bitmask_alloc(numa_num_configured_nodes());
        numa_bitmask_setbit(mask, numa_node);
        numa_bind(mask);
#else
        numa_node = 0;
#endif

        steal_from = thread_id_to_steal_from[thread_id];
        steal_to = thread_id_to_steal_to[thread_id];

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);

        int cpuid = thread_id_to_core[thread_id];

        CPU_SET(cpuid, &cpuset);
        sched_setaffinity(gettid(), sizeof(cpuset), &cpuset);
#ifdef USE_NUMA
printf("thread_id = %d, nodes = %d, thread_num = %d, numa_node = %d, cpuid = %d, steal_from = %d, steal_to = %d\n", thread_id, numa_num_configured_nodes(), thread_num_, numa_node, cpuid, steal_from, steal_to);
#else
printf("thread_id = %d, thread_num = %d, cpuid = %d, steal_from = %d, steal_to = %d\n", thread_id, thread_num_, cpuid, steal_from, steal_to);
#endif
fflush(stdout);
    }

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
#if 0
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
#endif // work steal
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
#if !use_epoll && !use_poll
            if (++idle > worker_thread_idle_threshold) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            } else {
                #if use_yield
                sched_yield();
                #endif
            }
#else
            uint64_t foo;
            #if use_poll
            struct pollfd evt = { .fd = evfd[thread_id], .events = POLLIN };
            int n = poll(&evt, 1, 2 * 1000);
            #else
            struct epoll_event evt;
            int n = epoll_wait(epfd[thread_id], &evt, 1, 3 * 1000);
            #endif
            if (n == 1) {
                read(evfd[thread_id], &foo, 8);
            }
#endif
        } else if (status == ThreadStatus::EXIT) {
            return;
        }
    }
}
