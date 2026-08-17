/**
 * @Description  : Persistent per-NUMA hugetlbfs backing for resident MoE
 *                 expert weights (BufferB). Weights live in one growing file
 *                 per NUMA node under a hugetlbfs mount; small ".done" marker
 *                 files on the regular filesystem record validity so that a
 *                 restarted process can mmap the already-resident hugepages
 *                 and skip reloading from safetensors.
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#ifndef CPUINFER_CPU_BACKEND_HUGEPAGE_WEIGHTS_H
#define CPUINFER_CPU_BACKEND_HUGEPAGE_WEIGHTS_H

#include <fcntl.h>
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>

#ifndef HUGETLBFS_MAGIC
#define HUGETLBFS_MAGIC 0x958458f6
#endif

namespace hugepage_weights {

constexpr const char* kMarkerMagic = "KTHP1";

inline uint64_t fnv1a64(const void* data, size_t len, uint64_t h = 14695981039346656037ULL) {
  const uint8_t* p = (const uint8_t*)data;
  for (size_t i = 0; i < len; i++) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

inline uint64_t fnv1a64_str(const std::string& s) { return fnv1a64(s.data(), s.size()); }

inline std::string data_root() {
  const char* e = getenv("KT_HUGEPAGE_WEIGHT_DIR");
  return (e && *e) ? e : "/dev/hugepages/kt_weights";
}

inline std::string meta_root() {
  const char* e = getenv("KT_HUGEPAGE_WEIGHT_META_DIR");
  return (e && *e) ? e : "/var/lib/kt-hugepage-weights";
}

inline std::string env_or_empty(const char* name) {
  const char* e = getenv(name);
  return (e && *e) ? e : "";
}

// One allocation handed out by alloc(). `ptr` is a slice of the per-NUMA
// arena mapping; `reused` means the marker fingerprint matched, i.e. the
// bytes already contain valid converted weights for this exact layout.
struct Segment {
  uint8_t* ptr = nullptr;
  size_t offset = 0;
  size_t size = 0;
  int numa = -1;
  bool reused = false;
  uint64_t fp = 0;      // C++-side fingerprint (layout + stamp + backend tag)
  uint64_t pfp = 0;     // python-visible fingerprint (checked by kt_kernel python)
  std::string map_line; // committed physical_to_logical fingerprint, e.g. "map=0123abcd..."
  std::string key;
};

namespace detail {

// Fixed virtual bases per NUMA node. Weights must stay at stable addresses
// across arena growth: layers already constructed keep raw pointers into the
// mapping, so every remap re-uses the same fixed address (old range is
// munmapped first, then MAP_FIXED re-maps the grown file at the same base).
constexpr uintptr_t kBaseNode0 = 0x200000000000ULL;
constexpr uintptr_t kNodeStride = 0x10000000000ULL;

inline uintptr_t fixed_base(int node) { return kBaseNode0 + (uintptr_t)node * kNodeStride; }

struct Arena {
  int fd = -1;
  int node = -1;
  uint8_t* base = nullptr;
  size_t map_len = 0;   // mmap length (hugepage-rounded)
  size_t file_size = 0; // logical end of committed data (ftruncate value)
  size_t page = 1 << 20;
  size_t cursor = 0;    // next free offset, driven by marker validation order
};

inline std::mutex& mu() {
  static std::mutex m;
  return m;
}

inline std::map<int, Arena>& arenas() {
  static std::map<int, Arena> a;
  return a;
}

inline size_t round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

struct Marker {
  bool ok = false;
  size_t offset = 0;
  size_t size = 0;
  uint64_t fp = 0;
  uint64_t pfp = 0;
  std::string map_line;
};

inline Marker read_marker(const std::string& path) {
  Marker m;
  std::ifstream f(path);
  if (!f.is_open()) return m;
  std::string line;
  if (!std::getline(f, line) || line != kMarkerMagic) return m;
  while (std::getline(f, line)) {
    auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string k = line.substr(0, eq), v = line.substr(eq + 1);
    if (k == "offset") m.offset = strtoull(v.c_str(), nullptr, 10);
    else if (k == "size") m.size = strtoull(v.c_str(), nullptr, 10);
    else if (k == "fp") m.fp = strtoull(v.c_str(), nullptr, 16);
    else if (k == "pfp") m.pfp = strtoull(v.c_str(), nullptr, 16);
    else if (k == "map") m.map_line = "map=" + v;
  }
  m.ok = true;
  return m;
}

}  // namespace detail

// Must be called at least once per process before alloc(); cheap.
inline bool enabled() {
  const char* e = getenv("KT_HUGEPAGE_WEIGHTS");
  if (e && e[0] == '0') return false;
  std::error_code ec;
  const std::string root = data_root();
  std::filesystem::create_directories(root, ec);
  if (ec) return false;
  struct statfs st {};
  if (statfs(root.c_str(), &st) != 0) return false;
  if ((uint32_t)st.f_type != (uint32_t)HUGETLBFS_MAGIC) {
    fprintf(stderr,
            "[hugepage_weights] %s is not on hugetlbfs (fstype=%#x); "
            "resident weights fall back to heap allocation.\n",
            root.c_str(), (unsigned)st.f_type);
    return false;
  }
  return true;
}

// `bytes` must be a multiple of 64. `fp`/`pfp` are caller-computed layout
// fingerprints; the physical_to_logical map hash is validated separately in
// check/commit so a changed expert map only invalidates the marker.
inline Segment alloc(int numa, const std::string& key, size_t bytes, uint64_t fp, uint64_t pfp) {
  Segment seg;
  seg.numa = numa;
  seg.key = key;
  seg.size = bytes;
  seg.fp = fp;
  seg.pfp = pfp;
  if (!enabled() || bytes == 0) return seg;

  std::lock_guard<std::mutex> lk(detail::mu());
  auto& a = detail::arenas()[numa];
  if (a.fd < 0) {
    a.node = numa;
    const std::string node_dir = data_root() + "/node" + std::to_string(numa);
    const std::string meta_dir = meta_root() + "/node" + std::to_string(numa);
    std::error_code ec;
    std::filesystem::create_directories(node_dir, ec);
    std::filesystem::create_directories(meta_dir, ec);
    if (ec) {
      fprintf(stderr, "[hugepage_weights] cannot create %s: %s\n", node_dir.c_str(), ec.message().c_str());
      a.fd = -2; // mark permanently broken
      return seg;
    }
    struct statfs st {};
    statfs(node_dir.c_str(), &st);
    if (st.f_bsize > 0) a.page = st.f_bsize;
    a.fd = open((node_dir + "/weights.bin").c_str(), O_RDWR | O_CREAT, 0644);
    if (a.fd < 0) {
      fprintf(stderr, "[hugepage_weights] open %s/weights.bin failed: %s\n", node_dir.c_str(), strerror(errno));
      a.fd = -2;
      return seg;
    }
    struct stat fst {};
    fstat(a.fd, &fst);
    a.file_size = (size_t)fst.st_size;
    if (a.file_size > 0) {
      a.map_len = detail::round_up(a.file_size, a.page);
      a.base = (uint8_t*)mmap((void*)detail::fixed_base(numa), a.map_len, PROT_READ | PROT_WRITE,
                              MAP_SHARED | MAP_FIXED_NOREPLACE, a.fd, 0);
      if (a.base == MAP_FAILED) {
        // Fixed address busy: try an OS-chosen address. If it differs from
        // any previous mapping this process used, later growth cannot move
        // it, so disable the arena instead of risking dangling pointers.
        a.base = nullptr;
        a.map_len = 0;
        fprintf(stderr, "[hugepage_weights] fixed map at %p busy; node %d arena disabled this run\n",
                (void*)detail::fixed_base(numa), numa);
        a.fd = -2;
        return seg;
      }
      unsigned long nodemask = 1UL << numa;
      mbind(a.base, a.map_len, MPOL_BIND, &nodemask, sizeof(nodemask) * 8, 0);
    }
    // cursor starts at 0: validity of pre-existing content is proven layer by
    // layer through marker fingerprints, in deterministic load order.
    a.cursor = 0;
    printf("[hugepage_weights] node %d arena opened: file=%zu bytes, hugepage=%zu KiB\n", numa, a.file_size,
           a.page / 1024);
  }
  if (a.fd < 0) return seg;
  if (a.base == nullptr && a.file_size > 0) return seg; // mmap previously failed

  const std::string marker_path = meta_root() + "/node" + std::to_string(numa) + "/" + key + ".done";
  detail::Marker m = detail::read_marker(marker_path);
  if (m.ok && m.offset == a.cursor && m.size == bytes && m.fp == fp && m.pfp == pfp &&
      a.file_size >= m.offset + m.size) {
    seg.ptr = a.base + m.offset;
    seg.offset = m.offset;
    seg.reused = true;
    seg.map_line = m.map_line;
    a.cursor = m.offset + m.size;
    return seg;
  }

  // Fresh (or invalid) segment: carve at cursor and extend the file as needed.
  size_t off = a.cursor;
  size_t need = detail::round_up(off + bytes, a.page);
  if (need > a.file_size) {
    if (ftruncate(a.fd, (off_t)need) != 0) {
      fprintf(stderr, "[hugepage_weights] ftruncate to %zu failed: %s\n", need, strerror(errno));
      return seg;
    }
    a.file_size = need;
  }
  if (a.base == nullptr || a.map_len < a.file_size) {
    // Grow the mapping. The old range is released first and the grown file is
    // re-mapped at the SAME fixed address so pointers handed out for earlier
    // layers stay valid (contents are identical, file-backed, same offsets).
    size_t new_len = a.file_size;
    uint8_t* new_base = (uint8_t*)mmap((void*)(a.base != nullptr ? (uintptr_t)a.base : (uintptr_t)detail::fixed_base(numa)),
                                       new_len, PROT_READ | PROT_WRITE,
                                       MAP_SHARED | MAP_FIXED | (a.base != nullptr ? 0 : MAP_FIXED_NOREPLACE), a.fd, 0);
    if (new_base == MAP_FAILED) {
      fprintf(stderr, "[hugepage_weights] mmap %zu failed: %s\n", new_len, strerror(errno));
      a.file_size = off; // roll back logical size; arena unusable this run
      ftruncate(a.fd, (off_t)off);
      return seg;
    }
    a.base = new_base;
    a.map_len = new_len;
    unsigned long nodemask = 1UL << numa;
    mbind(a.base, a.map_len, MPOL_BIND, &nodemask, sizeof(nodemask) * 8, 0);
  }
  seg.ptr = a.base + off;
  seg.offset = off;
  a.cursor = off + bytes;
  return seg;
}

// Persist the validity marker after converted weights have been fully written
// into the segment. `map_hash` fingerprints the physical_to_logical map the
// content was laid out with.
inline void commit(Segment& seg, uint64_t map_hash) {
  if (seg.ptr == nullptr) return;
  char map_line[48];
  snprintf(map_line, sizeof(map_line), "map=%016llx", (unsigned long long)map_hash);
  std::ostringstream ss;
  ss << kMarkerMagic << "\n"
     << "offset=" << seg.offset << "\n"
     << "size=" << seg.size << "\n"
     << "fp=" << std::hex << seg.fp << std::dec << "\n"
     << "pfp=" << std::hex << seg.pfp << std::dec << "\n"
     << map_line << "\n";
  const std::string path = meta_root() + "/node" + std::to_string(seg.numa) + "/" + seg.key + ".done";
  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::trunc);
    if (!f.is_open()) {
      fprintf(stderr, "[hugepage_weights] cannot write marker %s\n", tmp.c_str());
      return;
    }
    f << ss.str();
  }
  seg.map_line = map_line;
  seg.reused = true; // now valid for future runs
  // Data pages of a hugetlbfs file are always resident; syncing the marker is
  // what makes the segment durable against process crashes.
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec) fprintf(stderr, "[hugepage_weights] rename marker failed: %s\n", ec.message().c_str());
}

}  // namespace hugepage_weights

#endif  // CPUINFER_CPU_BACKEND_HUGEPAGE_WEIGHTS_H
