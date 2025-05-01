# KTransformers Optimized

在KTransformers的基础上进行了一些优化。

## 注意点

1. 运行前需要编译`core_info.c`：`cd custom && make && cd ../`
1. 必须cd到ktransformers工程所在路径再运行（和官方文档一致，只不过因为代码里写死了动态链接库的相对路径，所以这一项变成了强制要求）
1. 只支持Linux
1. 优化主要在`USE_NUMA`的基础上进行，只有持久化巨页在non-NUMA上做了实现
1. 因为平台固定下来后，最优性能配置不会变化，所以目前代码写得比较死，比如`core_info.c`编译为动态链接库而非单独的配置文件
1. 模型内存会使用巨页并被持久化，这样进程可以快速启动。请确保巨页文件系统已挂载到`/dev/hugepages`（ubuntu默认就会挂载到这个位置）并且确保启动ktransformers的用户有权限操作该路径（`sudo chmod 777 /dev/hugepages`）
1. KT0.3版本开始需要手动指定kvcache占用显存的大小，这里增加了一个参数`--gpu_memory_size`，目前配置比较麻烦，见`gpu_memory_size`一节

## 配置

### core\_info.c

位于`custom/core_info.c`。

可使用脚本`./scripts/generate-core-info.py`生成`core_info.c`。  
脚本默认会为`cpu 0`预留末尾的3个核心，同时占用`cpu 1`的所有核心，用于非`llama.cpp`线程的处理。

如果你的GPU位于`numa1`上，则可以指定参数，为`cpu 1`预留3个核心而非`cpu 0`：`--gpu-numa=1`。

core\_info.h也可以手动配置。

* `worker_thread_idle_threshold`: cpu worker空循环多少轮后进入sleep；建议设置为`CPU Hz数 / 10`
* `thread_id_to_numa`: 线程id到numa id的映射，用于绑定numa。其数组下标为thread\_id，从0开始，到`--cpu_infer - 2`结束（`--cpu_infer - 1`是`llama.cpp`线程的数量）
* `thread_id_to_core`: 线程id到cpu core id的映射，用于绑核。
* `prefill_core_enabled`: 在prefill阶段，指定线程是否启用
* `decode_core_enabled`: 在decode阶段，指定线程是否启用
* `thread_id_to_steal_from`: 最新提交可以忽略，但如果开启work steal，则需要配置。该项指每个线程应当从哪个线程id开始steal work
* `thread_id_to_steal_to`: 最新提交可以忽略，但如果开启work steal，则需要配置。该项指每个线程的work steal到哪个线程id为止（不包括指定的线程id）

修改后，只需编译该c文件为动态链接库即可，不需要完整编译ktransformers。

编译方式：`cd custom && make && cd ..`

注意，加载动态链接库使用了相对路径。启动ktransformers时，需要cd到ktransformers工程所在的路径。

### /tmp/kt\_per\_numa\_huge\_mem

非必选。

每个numa分配的巨页大小，单位为字节。默认值写死在代码里，是`375G`巨页（适配于Q4，而Q8每个numa需要不到`650G`）

```shell
echo 697932185600 > /tmp/kt_per_numa_huge_mem
```

### /tmp/kt\_force\_think\_prefix

非必选。

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

为了让当前用户可以操作巨页文件系统，可以把`/dev/hugepages`权限设置为`777`：

```shell
sudo chmod 777 /dev/hugepages
```

## 运行

在上述配置完成，正确重启机器，编译安装好ktransformers之后，执行：

```shell
numactl --cpunodebind=0 --interleave=0 \
env LD_PRELOAD=${libmimalloc.so的完整路径} MIMALLOC_VERBOSE=1 MIMALLOC_ALLOW_LARGE_OS_PAGES=1 \
python ktransformers/server/main.py \
	--architectures ${DeepseekV3ForCausalLM} \
	--model_path ${模型元数据路径} \
	--gguf_path  ${gguf文件所在目录} \
	--optimize_config_path ktransformers/optimize/optimize_rules/DeepSeek-V3-Chat-serve.yaml \
	--cpu_infer  ${系统总核心数 + 1 - 3} \
	--gpu_memory_size 2147483648 \
	--max_new_tokens 28000 \
	--cache_lens     28000 \
	--chunk_size     64 \
	--max_batch_size 4 \
	--backend_type balance_serve \
	--force_think \
	--web False
```

如果你的GPU位于`numa 1`上，那么`--cpunodebind=0 --interleave=0`调整为`--cpunodebind=1 --interleave=1`。  
如果你是手动修改的cpu\_info.c，那么`--cpu_infer`需要相应进行设置。注意`cpu_infer - 1`才是CPU worker线程数。   

## gpu\_memory\_size和cache\_lens的取值

这个选项用于控制rpc进程的显存占用。现在这个配置取值比较麻烦，需要首次对话后才能够确定配置是否可用。  
配合本优化分支的持久巨页倒是好很多（重启时间基本在半分钟内），而官方版本调试起来就很折磨。。。

首先随便选一个比较小的数值，比如2147483648（2G），启动KT。

启动后使用`nvidia-smi`查看显存余量，如果还有较多余量，则修改`--gpu_memory_size`直到刚好占满显卡（稍微留一点余量）。

我们先把`--max_new_tokens`和`--cache_lens`调成一样的数值，然后进行一次对话，并查看rpc.log：`~/.ktransformers/logs/rpc.log`，里面会有`!!`开头的日志：

```
!! Before gpu_only_alloc_col: estimated_length={} NumTokenPerBlock={} total_block_count={}
!! GPUPageCache::gpu_only_alloc_col: count={} total_kvcache_pages={} actual_size={}
```

主要看`GPUPageCache::gpu_only_alloc_col`这一行，其中`count=`为所需的pages数量，`total_kvcache_pages=`可用的pages数量。可用数量是根据`--gpu_memory_size`确定的。前一步已经调好了。

如果这里`count > total_kvcache_pages`，那么将无法进行推理。这个`count`是根据输入token数量和`max_new_tokens`确定的，如果count特别大，则调低`--max_new_tokens`和`--cache_lens`，如果`count`比`total_kvcache_pages`小很多，则调高`count`。用简单的比例关系算一下应该调到多少。

调整完成后，可以适当调低`max_new_tokens`，以实现多并发。如果不需要多并发则直接使用即可。

## 观测

### numa-stats.sh

每2秒刷新一次`numastat -n`，统计值展示为 per second 的数据。

### show-cpu.sh

显示当前每个CPU核的频率，建议配合`watch`使用。

### show-mem.sh

显示每个numa上的`2M`、`1G`巨页，以及全局的匿名巨页，建议配合`watch`使用。
