#!/usr/bin/env python3

import sys
import os
import argparse
import json

def build_socket_id_range(socket, gpu_numa):
    r = range(0, len(socket))
    if gpu_numa == 1:
        r = reversed(r)
    return r

def print_tid_comment(content, tid, gpu_numa, socket_topo, s):
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
    for c in cpu_topo:
        with open(f'/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list') as f:
            siblings = f.read().strip().split(',')
            if len(siblings) > 2:
                print(f'Invalid thread siblings list {siblings}, expecting at most 2 elements')
                return 1
            if len(siblings) == 0:
                print(f'Invalid thread siblings list {siblings}, no thread exists')
                return 1
            if len(siblings) == 2:
                cpu_topo[c]['smt'] = {
                    int(siblings[0].strip()): int(siblings[1].strip()),
                    int(siblings[1].strip()): int(siblings[0].strip()),
                }
            else:
                cpu_topo[c]['smt'] = { int(siblings[0].strip()): int(siblings[0].strip()) }

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
    print('---------')
    print()

    content = ''
    tid = 0
    content += 'static int _kt_thread_id_to_core[] = {\n'
    for s in build_socket_id_range(socket_topo, gpu_numa):
        cores = socket_topo[s]['cores']
        content, tid = print_tid_comment(content, tid, gpu_numa, socket_topo, s)
        content += '   '
        for c in cores:
            if cores.index(c) >= len(cores) - 3:
                if gpu_numa == s:
                    continue
            content += f'{c}'
            content += ','
            if c < 10:
                content += ' '
        content += '\n'
    content += '};\n' # _kt_thread_id_to_core

    content += '\n'
    tid = 0
    content += 'static int _kt_prefill_core_enabled[] = {\n'
    for s in build_socket_id_range(socket_topo, gpu_numa):
        cores = socket_topo[s]['cores']
        content, tid = print_tid_comment(content, tid, gpu_numa, socket_topo, s)
        content += '   '
        for c in cores:
            if cores.index(c) >= len(cores) - 3:
                if gpu_numa == s:
                    continue
            if cpu_topo[c]['smt'][c] < c:
                content += '0'
            else:
                content += '1'
            content += ', '
        content += '\n'
    content += '};\n' # _kt_prefill_core_enabled

    content += '\n'
    tid = 0
    content += 'static int _kt_decode_core_enabled[] = {\n'
    for s in build_socket_id_range(socket_topo, gpu_numa):
        cores = socket_topo[s]['cores']
        content, tid = print_tid_comment(content, tid, gpu_numa, socket_topo, s)
        content += '   '
        for c in cores:
            if cores.index(c) >= len(cores) - 3:
                if gpu_numa == s:
                    continue
            content += '1, '
        content += '\n'
    content += '};\n' # _kt_decode_core_enabled

    content += '\n'
    content += 'long kt_worker_thread_idle_threshold() {\n'
    content += f'    return {int(freq / 10)}; // cpu Hz / 10\n'
    content += '}\n'

    content += '\n'
    content += 'int kt_thread_id_to_numa(int thread_id) {\n'
    beg_tid = 0
    for s in build_socket_id_range(socket_topo, gpu_numa):
        cores = socket_topo[s]['cores']
        end_tid = beg_tid + len(cores) - 1
        if beg_tid == 0:
            end_tid -= 3
        content += '    if (' + str(beg_tid) + ' <= thread_id && thread_id <= ' + str(end_tid) + ') {\n'
        content += '        return ' + str(s) + ';\n'
        content += '    }\n'
        beg_tid = end_tid + 1
    content += '    return 0; // should not reach here\n'
    content += '}\n' # kt_thread_id_to_numa

    content += '\n'
    content += '''
int kt_thread_id_to_core(int thread_id) {
    return _kt_thread_id_to_core[thread_id];
}

int kt_prefill_core_enabled(int thread_id) {
    return _kt_prefill_core_enabled[thread_id];
}

int kt_decode_core_enabled(int thread_id) {
    return _kt_decode_core_enabled[thread_id];
}
'''.strip()
    content += '\n'

    content += '\n'
    content += 'int kt_thread_id_to_steal_from(int thread_id) {\n'
    beg_tid = 0
    for s in build_socket_id_range(socket_topo, gpu_numa):
        cores = socket_topo[s]['cores']
        end_tid = beg_tid + len(cores) - 1
        if beg_tid == 0:
            end_tid -= 3
        content += '    if (' + str(beg_tid) + ' <= thread_id && thread_id <= ' + str(end_tid) + ') {\n'
        content += '        return ' + str(beg_tid) + ';\n'
        content += '    }\n'
        beg_tid = end_tid + 1
    content += '    return 0; // should not reach here\n'
    content += '}\n' # kt_thread_id_to_steal_from

    content += '\n'
    content += 'int kt_thread_id_to_steal_to(int thread_id) {\n'
    beg_tid = 0
    for s in build_socket_id_range(socket_topo, gpu_numa):
        cores = socket_topo[s]['cores']
        end_tid = beg_tid + len(cores) - 1
        if beg_tid == 0:
            end_tid -= 3
        content += '    if (' + str(beg_tid) + ' <= thread_id && thread_id <= ' + str(end_tid) + ') {\n'
        content += '        return ' + str(end_tid + 1) + ';\n'
        content += '    }\n'
        beg_tid = end_tid + 1
    content += '    return ' + str(beg_tid) + '; // should not reach here\n'
    content += '}\n' # kt_thread_id_to_steal_to

    print(content)

    CORE_INFO_C_PATH = './custom/core_info.c'
    if not os.path.exists(CORE_INFO_C_PATH):
        print(f'{CORE_INFO_C_PATH} does not exist. Please run this script on the root of the git directory')
        return 1
    with open(CORE_INFO_C_PATH, 'w') as f:
        f.write(content)

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-numa', type=int, default=0, help="numa position of the gpu")
    args = parser.parse_args()
    sys.exit(main(args))
