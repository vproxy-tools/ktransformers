#ifndef _CORE_INFO_H_
#define _CORE_INFO_H_

#include <dlfcn.h>

static void* libcore_info = nullptr;

static long worker_thread_idle_threshold = 0;

typedef int (*fn_thread_id_to_numa)(int thread_id);
static fn_thread_id_to_numa thread_id_to_numa = nullptr;

typedef int (*fn_thread_id_to_core)(int thread_id);
static fn_thread_id_to_core thread_id_to_core = nullptr;

typedef int (*fn_prefill_core_enabled)(int thread_id);
static fn_prefill_core_enabled prefill_core_enabled = nullptr;

typedef int (*fn_decode_core_enabled)(int thread_id);
static fn_decode_core_enabled decode_core_enabled = nullptr;

typedef int (*fn_thread_id_to_steal_from)(int thread_id);
static fn_thread_id_to_steal_from thread_id_to_steal_from = nullptr;

typedef int (*fn_thread_id_to_steal_to)(int thread_id);
static fn_thread_id_to_steal_to thread_id_to_steal_to = nullptr;

typedef int (*fn_work_stealing_enabled)(int thread_id);
static fn_work_stealing_enabled work_stealing_enabled = nullptr;

static inline void open_dylib() {
    if (libcore_info == nullptr) {
        void* handle = dlopen("./custom/libcore_info.so", RTLD_NOW | RTLD_LOCAL);
        if (!handle) {
            fprintf(stderr, "Error: %s\n", dlerror());
            exit(1);
        }

        void* sym = dlsym(handle, "kt_worker_thread_idle_threshold");
        if (sym == nullptr) {
            fprintf(stderr, "cannot find kt_worker_thread_idle_threshold\n");
            exit(1);
        }
        typedef long (*fn_worker_thread_idle_threshold)();
        worker_thread_idle_threshold = ((fn_worker_thread_idle_threshold)sym)();

        sym = dlsym(handle, "kt_thread_id_to_numa");
        if (sym == nullptr) {
            fprintf(stderr, "cannot find kt_thread_id_to_numa\n");
            exit(1);
        }
        thread_id_to_numa = (fn_thread_id_to_numa)sym;

        sym = dlsym(handle, "kt_thread_id_to_core");
        if (sym == nullptr) {
            fprintf(stderr, "cannot find kt_thread_id_to_core\n");
            exit(1);
        }
        thread_id_to_core = (fn_thread_id_to_core)sym;

        sym = dlsym(handle, "kt_prefill_core_enabled");
        if (sym == nullptr) {
            fprintf(stderr, "cannot find kt_prefill_core_enabled\n");
            exit(1);
        }
        prefill_core_enabled = (fn_prefill_core_enabled)sym;

        sym = dlsym(handle, "kt_decode_core_enabled");
        if (sym == nullptr) {
            fprintf(stderr, "cannot find kt_decode_core_enabled\n");
            exit(1);
        }
        decode_core_enabled = (fn_decode_core_enabled)sym;

        sym = dlsym(handle, "kt_thread_id_to_steal_from");
        if (sym == nullptr) {
            fprintf(stderr, "cannot find kt_thread_id_to_steal_from\n");
            exit(1);
        }
        thread_id_to_steal_from = (fn_thread_id_to_steal_from)sym;

        sym = dlsym(handle, "kt_thread_id_to_steal_to");
        if (sym == nullptr) {
            fprintf(stderr, "cannot find kt_thread_id_to_steal_to\n");
            exit(1);
        }
        thread_id_to_steal_to = (fn_thread_id_to_steal_to)sym;

        sym = dlsym(handle, "kt_work_stealing_enabled");
        if (sym == nullptr) {
            fprintf(stderr, "cannot find kt_work_stealing_enabled");
            exit(1);
        }
        work_stealing_enabled = (fn_work_stealing_enabled)sym;

        libcore_info = handle;
    }
}

#endif // _CORE_INFO_H_
