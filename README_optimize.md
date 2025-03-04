# KTransformers Optimized

在KTransformers的基础上进行了一些优化。

## 注意点

1. 只支持Linux
2. 优化主要在`USE_NUMA`的基础上进行，只有持久化巨页在non-NUMA上做了实现
3. 因为平台固定下来后，最优性能配置不会变化，所以目前代码写得比较死，比如`core_info.h`会编译到代码里而非单独的配置文件
4. 模型内存会使用巨页并被持久化，这样进程可以快速启动。请确保巨页文件系统已挂载到`/dev/hugepages`（ubuntu默认就会挂载到这个位置）并且确保启动ktransformers的用户有权限操作该路径

## 配置

### core\_info.h

位于`ktransformers/ktransformers_ext/cpu_backend/core_info.h`。

可使用脚本`./scripts/generate-core-info.py`生成`core_info.h`。  
脚本默认会为`cpu 0`预留末尾的3个核心，同时占用`cpu 1`的所有核心，用于非`llama.cpp`线程的处理。

如果你的GPU位于`numa1`上，则建议为`cpu 1`预留3个核心：`--gpu-numa=1`。

core\_info.h也可以手动配置。

* `worker_thread_idle_threshold`: llama.cpp worker空循环多少轮后进入sleep；建议设置为`CPU Hz数 / 10`
* `thread_id_to_numa`: 线程id到numa id的映射，用于绑定numa。其数组下标为thread\_id，从0开始，到`--cpu_infer - 2`结束（`--cpu_infer - 1`是`llama.cpp`线程的数量）
* `thread_id_to_core`: 线程id到cpu core id的映射，用于绑核。
* `thread_id_to_steal_from`: 最新提交可以忽略，但如果开启work steal，则需要配置。该项指每个线程应当从哪个线程id开始steal work
* `thread_id_to_steal_to`: 最新提交可以忽略，但如果开启work steal，则需要配置。该项指每个线程的work steal到哪个线程id为止（不包括指定的线程id）

### /tmp/kt\_per\_numa\_huge\_mem

每个numa分配的巨页大小，单位为字节。默认值写死在代码里，是`375G`巨页（适配于Q4，而Q8每个numa需要不到`650G`）

```shell
echo 697932185600 > /tmp/kt_per_numa_huge_mem
```

### /tmp/kt\_force\_think\_prefix

开启`--force_think`后才会生效。

在KTransformers添加的`<think>\n`标签后再增加指定的字符串。

例如：

```shell
echo '嗯，关于用户的这个问题，我应当按照指定的格式直接回答。' > /tmp/kt_force_think_prefix
echo '</think>' >> /tmp/kt_force_think_prefix
```

### batch yield

每当输出文本总长度超过阈值时才执行`yield`（会打印以及给客户端发送响应）。

修改文件：`ktransformers/server/backend/interfaces/transformers.py`，搜索`YIELD_THRESHOLD`修改即可。

## 系统设置

### 1. 开启巨页

先确认系统支持的巨页大小：

```shell
ls /sys/kernel/mm/hugepages/
```

然后进行配置：

```shell
sudo vim /etc/default/grub
```

在`GRUB_CMDLINE_LINUX`一项中，添加：

* `default_hugepagesz=1G`: 默认使用`1G`巨页。如果你的平台支持`2G`巨页那就写2G，越大越好
* `hugepagesz=1G hugepages=1400`: 配置启动时分配的`1G`巨页大小，本示例会分配`1400`个`1G`巨页
* `hugepagesz=2M hugepages=16384`: 配置启动是分配的`2M`巨页大小，本示例会分配`16384`个`2M`巨页，`2M`巨页会被`MIMALLOC`使用。多种巨页的配置可以同时出现
* `transparent_hugepage=never`: 禁止透明巨页，主要是为了方便观测每个numa上的剩余巨页数量，也可以不加这个选项

配置完成后，执行：

```shell
sudo update-grub
```

重启后生效。

### 2. 配置核隔离

这一步可选，理论上应该不会有影响，只是保险起见配一下。  
为了防止被调度到worker核上，可以配置一下核隔离。

```shell
sudo vim /etc/default/grub
```

在`GRUB_CMDLINE_LINUX`一项中，添加：

* `isolcpus=0-20,24-47`

“不想让linux自动调度的核”配置到这里即可。

配置完成后，执行：

```shell
sudo update-grub
```

重启后生效。

### 3. 编译MIMALLOC

```shell
git clone https://github.com/microsoft/mimalloc

# 按照readme.md文档编译即可

cd mimalloc
mkdir -p out/release
cd out/release
cmake ../..
make
```

记录`libmimalloc.so`的位置，后面会用到。

### 4. 禁用超线程

在BIOS中禁用超线程，一般叫`SMT`。

### 5. 将cpu全部设置为性能模式

每次开机都要设置，建议写个启动脚本，在ktransformers启动前执行一下。

```shell
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo "performance" | sudo tee $cpu 1>/dev/null
done
```

### 6. 配置巨页文件系统权限

为了让当前用户可以操作巨页文件系统，可以把`/dev/hugepages`的owner改为当前用户：

```shell
sudo chown ${当前用户} /dev/hugepages
```

## 运行

在上述配置完成，正确重启机器，编译安装好ktransformers之后，执行：

```shell
env LD_PRELOAD=${libmimalloc.so的完整路径} MIMALLOC_VERBOSE=1 MIMALLOC_ALLOW_LARGE_OS_PAGES=1 \
numactl --interleave=0 \
ktransformers \
	--model_path ${模型元数据路径} \
	--gguf_path  ${gguf文件所在目录} \
	--cpu_infer  ${系统总核心数 + 1 - 3} \
	--max_new_tokens 32768 \
	--cache_lens     32768 \
	--force_think \
	--web False
```

如果你的GPU位于`numa 1`上，那么`--interleave=0`调整为`--interleave=1`。  
如果你是手动修改的cpu\_info.h，那么`--cpu_infer`需要相应设置。注意`cpu_infer - 1`才是`llama.cpp`线程数。   

## 观测

### numa-stats.sh

每2秒刷新一次`numastat -n`，统计值展示为 per second 的数据。

### show-cpu.sh

显示当前每个CPU核的频率，建议配合`watch`使用。

### show-mem.sh

显示每个numa上的`2M`、`1G`巨页，以及全局的匿名巨页，建议配合`watch`使用。
