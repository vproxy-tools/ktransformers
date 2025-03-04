#!/usr/bin/env python3

import sys
import os
import argparse
import json

def print_tid_comment(content, tid, gpu_numa, socket_topo, cores, s):
    content += '/* '
    cores = socket_topo[s]['cores']
    for c in cores:
        if cores.index(c) >= len(cores) - 3:
            if gpu_numa == s:
                continue
        content += f'{tid} '
        if tid < 10:
            content += ' '
        tid += 1
    content += '*/\n'
    return content, tid

def main(args):
    with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq') as f:
        freq = int(f.read()) * 1000

    with open('/proc/cpuinfo', 'r') as f:
        cpuinfo = f.read()

    cpulines = cpuinfo.split('\n')
    current_cpu = -1
    cpu_topo = {}
    for cpuline in cpulines:
        split = cpuline.split(':')
        if len(split) != 2:
            continue
        k = split[0].strip()
        v = split[1].strip()
        if k == 'processor':
            current_cpu = int(v)
        elif k == 'physical id':
            if current_cpu not in cpu_topo:
                cpu_topo[current_cpu] = {}
            cpu_topo[current_cpu]['socket'] = int(v)

    max_socket = -1
    for k in cpu_topo:
        v = cpu_topo[k]['socket']
        if v > max_socket:
            max_socket = v

    if max_socket > 2:
        print(f'Currently {max_socket} sockets not supported')
        return 1

    gpu_numa = args.gpu_numa
    if gpu_numa > max_socket:
        print(f'Unable to find numa {gpu_numa} for gpu')
        return 1

    for c in range(0, len(cpu_topo)):
        if c not in cpu_topo:
            print(f'Unable to find core {c}, core id not consistent?')
            return 1

    socket_topo = {}
    for c in cpu_topo:
        v = cpu_topo[c]['socket']
        if v not in socket_topo:
            socket_topo[v] = {'cores': []}
        socket_topo[v]['cores'].append(c)

    for s in range(0, len(socket_topo)):
        if s not in socket_topo:
            print(f'Unable to find socket {s}, socket id not consistent?')
            return 1

    for s in socket_topo:
        socket_topo[s]['cores'].sort()

    if len(socket_topo[gpu_numa]['cores']) < 4:
        print(f'Too few cores on socket {gpu_numa}')
        return 1

    print('collected data:')
    print(f'base frequency is {freq / 1000 / 1000 / 1000}GHz')
    print(f'{max_socket + 1} sockets')
    print(f'gpu on numa {gpu_numa}')
    print(f'cpu_topo = {json.dumps(cpu_topo)}')
    print(f'socket_topo = {json.dumps(socket_topo)}')

    content = ''
    content += '#ifndef _CORE_INFO_H_\n'
    content += '#define _CORE_INFO_H_\n'
    content += '\n'
    content += f'static long worker_thread_idle_threshold = {int(freq / 10)}; // cpu Hz / 10\n'
    content += '\n'
    content += 'static int thread_id_to_numa[] = {\n'
    tid = 0
    for s in range(0, len(socket_topo)):
        cores = socket_topo[s]['cores']
        content, tid = print_tid_comment(content, tid, gpu_numa, socket_topo, cores, s)
        content += '   '
        for c in cores:
            if cores.index(c) >= len(cores) - 3:
                if gpu_numa == s:
                    continue
            content += f'{s}'
            if cores.index(c) != len(cores) - 1 or s != len(socket_topo) - 1:
                content += ','
            content += ' '
        content += '\n'
    content += '};\n' # thread_id_to_numa
    content += '\n'
    content += 'static int thread_id_to_core[] = {\n'
    tid = 0
    for s in range(0, len(socket_topo)):
        cores = socket_topo[s]['cores']
        content, tid = print_tid_comment(content, tid, gpu_numa, socket_topo, cores, s)
        content += '   '
        for c in cores:
            if cores.index(c) >= len(cores) - 3:
                if gpu_numa == s:
                    continue
            content += f'{c}'
            if cores.index(c) != len(cores) - 1 or s != len(socket_topo) - 1:
                content += ','
            if c < 10:
                content += ' '
        content += '\n'
    content += '};\n' # thread_id_to_core
    content += '\n'
    content += 'static int thread_id_to_steal_from[] = {\n'
    tid = 0
    for s in range(0, len(socket_topo)):
        first = tid
        cores = socket_topo[s]['cores']
        content, tid = print_tid_comment(content, tid, gpu_numa, socket_topo, cores, s)
        content += '   '
        for c in cores:
            if cores.index(c) >= len(cores) - 3:
                if gpu_numa == s:
                    continue
            content += f'{first}'
            if cores.index(c) != len(cores) - 1 or s != len(socket_topo) - 1:
                content += ','
            if first < 10:
                content += ' '
        content += '\n'
    content += '};\n' # thread_id_to_steal_from
    content += '\n'
    content += 'static int thread_id_to_steal_to[] = {\n'
    tid = 0
    for s in range(0, len(socket_topo)):
        cores = socket_topo[s]['cores']
        content, tid = print_tid_comment(content, tid, gpu_numa, socket_topo, cores, s)
        last = tid
        content += '   '
        for c in cores:
            if cores.index(c) >= len(cores) - 3:
                if gpu_numa == s:
                    continue
            content += f'{last}'
            if cores.index(c) != len(cores) - 1 or s != len(socket_topo) - 1:
                content += ','
            if last < 10:
                content += ' '
        content += '\n'
    content += '};\n' # thread_id_to_steal_to
    content += '\n'
    content += '#endif // _CORE_INFO_H_\n'

    print(content)

    CORE_INFO_H_PATH = './ktransformers/ktransformers_ext/cpu_backend/core_info.h'
    if not os.path.exists(CORE_INFO_H_PATH):
        print(f'{CORE_INFO_H_PATH} does not exist. Please run this script on the root of the git directory')
        return 1
    with open(CORE_INFO_H_PATH, 'w') as f:
        f.write(content)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-numa', type=int, default=0, help="numa position of the gpu")
    args = parser.parse_args()
    sys.exit(main(args))
