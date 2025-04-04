"""
Date: 2024-11-07 07:02:20
LastEditors: djw
LastEditTime: 2024-12-10 08:48:32
"""

import torch
from torch import nn
import queue
import signal
import queue
from typing import AsyncIterable
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
import asyncio
import multiprocessing
import time
import torch.multiprocessing as mp
import random
import torch.distributed as dist
import zmq
import tempfile
from ktransformers.server.balance_serve.inference.forward_batch import ForwardBatchInput, ForwardBatchOutput

from ktransformers.server.config.config import Config
from ktransformers.models.custom_modeling_deepseek_v3 import KDeepseekV3ForCausalLM
from ktransformers.models.custom_modeling_deepseek_v2 import KDeepseekV2ForCausalLM
from ktransformers.server.balance_serve.inference.query_manager import QueryManager
from ktransformers.server.balance_serve.settings import sched_ext



def pad_num_tokens(num_tokens):
    return (num_tokens + 63) // 64 * 64

def deduplicate_and_sort(lst):
    return sorted(set(lst))
class ModelRunner:
    """A CudaGraphRunner runs the forward pass of a model with CUDA graph and torch.compile."""

    model: KDeepseekV3ForCausalLM
    input: ForwardBatchInput | list[ForwardBatchInput]
    output: ForwardBatchOutput
    
    def __init__(self, model = None, device = None, use_cuda_graph = False, max_decode_batch_size = 1, max_chunk_size = 4096, num_mini_batches: int = 1, page_size = 256):
        
        self.stream = torch.cuda.Stream(device=device)
        # 先注释掉
        self.model = model  # Compile and move model to the specified device
        self.device = device
        self.input = None
        self.features_buf = None
        self.output = None
        self.graph_memory_pool = None
        self.cuda_graphs = deduplicate_and_sort([1, 2, 3, Config().max_batch_size, 64, Config().chunk_size])
        self.use_cuda_graph = use_cuda_graph
        self.model_time = 0
        self.page_size = page_size
        # GPU timing for model execution
        self.start_model_event = torch.cuda.Event(enable_timing=True)
        self.end_model_event = torch.cuda.Event(enable_timing=True)
        if isinstance(self.cuda_graphs, list):
            self.graphs = [torch.cuda.CUDAGraph() for _ in range(len(self.cuda_graphs))]
            self.page_idx_buf = [torch.zeros([self.cuda_graphs[i]], dtype=torch.int32, device = self.device) for i in range(len(self.cuda_graphs))]
            self.page_offset_buf = [torch.zeros([self.cuda_graphs[i]], dtype=torch.int32, device = self.device) for i in range(len(self.cuda_graphs))]
        else:
            self.graphs = torch.cuda.CUDAGraph()
            self.page_idx_buf = torch.zeros([self.cuda_graphs], dtype=torch.int32, device = self.device)
            self.page_offset_buf = torch.zeros([self.cuda_graphs], dtype=torch.int32, device = self.device)
        self.num_mini_batches = num_mini_batches

        self.max_chunk_size = max_chunk_size

        self.bsz_tensor_buf = torch.empty((1, ),dtype=torch.int32, device=device)
        self.num_tokens_tensor_buf = torch.empty((1, ),dtype=torch.int32, device=device)
    def warmup(self):

        def capture_graphs(cuda_graph_idx=-1):
            if cuda_graph_idx != -1:
                with torch.cuda.graph(self.graphs[cuda_graph_idx], pool=self.graph_memory_pool, stream=self.stream):
                    self.outputs_buf[cuda_graph_idx] = self.model(self.input[cuda_graph_idx], self.features_buf[cuda_graph_idx], self.bsz_tensor_buf, self.num_tokens_tensor_buf, self.page_idx_buf[cuda_graph_idx], self.page_offset_buf[cuda_graph_idx], cuda_graph_idx=cuda_graph_idx)   
                self.graph_memory_pool = self.graphs[cuda_graph_idx].pool()
            else:
                with torch.cuda.graph(self.graphs, pool=self.graph_memory_pool, stream=self.stream):
                    self.outputs_buf = self.model(self.input, self.features_buf, self.bsz_tensor_buf, self.num_tokens_tensor_buf, self.page_idx_buf, self.page_offset_buf)   
                self.graph_memory_pool = self.graphs.pool()

        if isinstance(self.cuda_graphs, list):
            self.input = []
            self.features_buf = []
            self.outputs_buf = []
            self.bsz_tensor_buf = torch.tensor([0], dtype=torch.int32, device=self.device)
            self.num_tokens_tensor_buf = torch.tensor([0], dtype=torch.int32, device=self.device)
            for i in range(len(self.cuda_graphs)):
                prefill_query_length = (self.cuda_graphs[i] - Config().max_decode_batch_size) // Config().max_prefill_batch_size if self.cuda_graphs[i] > Config().max_decode_batch_size else 0  #@TODO only supprot 2 prefill batch
                self.input.append(ForwardBatchInput.gen_max_forward_batch(device=self.device, num_mini_batches = self.num_mini_batches, prefill_query_length=prefill_query_length, prefill_active_length=prefill_query_length, page_size=self.page_size, cuda_lens = self.cuda_graphs[i]))

                self.features_buf.append(self.model.batch_embeddings(self.input[i]))
                batch_size = self.input[i].minibatch.q_indptr.size(0)-1
                num_tokens = self.features_buf[i][0].size(0)
                print("capturing cuda graph", batch_size, num_tokens)
                self.bsz_tensor_buf[0] = batch_size
                self.num_tokens_tensor_buf[0] = num_tokens

                self.model.flash_infer_attn_plan(self.input[i], self.bsz_tensor_buf, self.num_tokens_tensor_buf,
                                                num_heads=self.model.config.num_attention_heads, head_dim_ckv=self.model.config.kv_lora_rank, 
                                                    head_dim_kpe=self.model.config.qk_rope_head_dim, page_size=self.model.cache.page_size, causal=True,
                                                    sm_scale=self.model.model.layers[0].self_attn.softmax_scale, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
            
                page_idx, page_offset = self.model.cache.get_page_table(self.input[i].minibatch.position_ids, self.input[i].minibatch.q_indptr, self.input[i].minibatch.kv_indptr, self.input[i].minibatch.kv_indices, self.num_tokens_tensor_buf)

                self.page_idx_buf[i][:num_tokens].copy_(page_idx[:num_tokens])
                self.page_offset_buf[i][:num_tokens].copy_(page_offset[:num_tokens])
                self.page_idx_buf[i][num_tokens:].fill_(self.model.cache.max_cache_len // self.model.cache.page_size -1) 
                
                self.outputs_buf.append(None)
            
                torch.cuda.synchronize()
                for warm_up_iters in range(11):
                    with torch.cuda.stream(self.stream):
                        self.outputs_buf[i] = self.model(self.input[i], self.features_buf[i], self.bsz_tensor_buf, self.num_tokens_tensor_buf, self.page_idx_buf[i], self.page_offset_buf[i])
                torch.cuda.synchronize()

                capture_graphs(i)

                with torch.cuda.stream(self.stream):
                    self.graphs[i].replay()

                self.sync(calc_time=False)
                print(f"cuda_graph: {i+1}/{len(self.cuda_graphs)}, warmup finished.")
        else:
            self.input = ForwardBatchInput.gen_max_forward_batch(device=self.device, num_mini_batches = self.num_mini_batches)

            self.features_buf = self.model.batch_embeddings(self.input)
            batch_size = self.input.minibatch.q_indptr.size(0)-1
            num_tokens = self.features_buf[0].size(0)

            
            self.bsz_tensor_buf = torch.tensor([batch_size], dtype=torch.int32, device=self.device)
            self.num_tokens_tensor_buf = torch.tensor([num_tokens], dtype=torch.int32, device=self.device)


            self.model.flash_infer_attn_plan(self.input, self.bsz_tensor_buf, self.num_tokens_tensor_buf,
                                            num_heads=self.model.config.num_attention_heads, head_dim_ckv=self.model.config.kv_lora_rank, 
                                                head_dim_kpe=self.model.config.qk_rope_head_dim, page_size=self.model.cache.page_size, causal=True,
                                                sm_scale=self.model.model.layers[0].self_attn.softmax_scale, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
            
            page_idx, page_offset = self.model.cache.get_page_table(self.input.minibatch.position_ids, self.input.minibatch.q_indptr, self.input.minibatch.kv_indptr, self.input.minibatch.kv_indices, self.num_tokens_tensor_buf)
            self.page_idx_buf[:num_tokens].copy_(page_idx[:num_tokens])
            self.page_offset_buf[:num_tokens].copy_(page_offset[:num_tokens])
            self.page_idx_buf[num_tokens:].fill_(self.model.cache.max_cache_len // self.model.cache.page_size - 1) 
            
            
            torch.cuda.synchronize()
            for warm_up_iters in range(11):
                with torch.cuda.stream(self.stream):
                    self.outputs_buf = self.model(self.input, self.features_buf, self.bsz_tensor_buf, self.num_tokens_tensor_buf, self.page_idx_buf, self.page_offset_buf)
            torch.cuda.synchronize()

            def capture_graphs():
                with torch.cuda.graph(self.graphs,  stream=self.stream):
                    self.outputs_buf = self.model(self.input, self.features_buf, self.bsz_tensor_buf, self.num_tokens_tensor_buf, self.page_idx_buf, self.page_offset_buf)   
                    # self.graph_memory_pool = self.graphs.pool()


            capture_graphs()

            with torch.cuda.stream(self.stream):
                self.graphs.replay()

            self.sync(calc_time=False)
            print("warmup finished.")
        
    def run(self, batch: sched_ext.BatchQueryTodo = None, query_manager: QueryManager = None):
        with torch.cuda.stream(self.stream):

            batch_size = len(batch.prefill_mini_batches) # TODO: calc this
            num_tokens = 0
            for i in range(len(batch.decode_mini_batches)):
                batch_size += len(batch.decode_mini_batches[i])
                num_tokens += len(batch.decode_mini_batches[i])
                #print(f'decode_batch_i: {len(batch.decode_mini_batches[i])},')

            for i in range(len(batch.prefill_mini_batches)):
                num_tokens += batch.prefill_mini_batches[i][2]
                #print(f'prefill_batch_i: {batch.prefill_mini_batches[i][2]},')



            if isinstance(self.cuda_graphs, list):
                # cuda graph idx equal to min idx i in self.cuda_graphs, that self.cuda_graphs[i] > num_tokens
                cuda_graph_idx = next((i for i, token in enumerate(self.cuda_graphs) if token >= num_tokens), len(self.cuda_graphs))
                if cuda_graph_idx == len(self.cuda_graphs):
                    assert False, "num_tokens is too large"
            else:
                cuda_graph_idx = -1
    
            if self.use_cuda_graph:
                if cuda_graph_idx != -1:
                    self.input[cuda_graph_idx].fill(batch, query_manager, self.page_size)
                else:
                    self.input.fill(batch, query_manager, self.page_size)
            else:
                self.input = ForwardBatchInput(batch=batch, query_manager=query_manager, device=self.device)
                    

            if cuda_graph_idx != -1 and self.use_cuda_graph:
                self.features = self.model.batch_embeddings(self.input[cuda_graph_idx], device=self.device)
            else:
                self.features = self.model.batch_embeddings(self.input, device=self.device)
                

            self.bsz_tensor_buf.copy_(batch_size)
            self.num_tokens_tensor_buf.copy_(torch.tensor([num_tokens], dtype=torch.int32, device=self.device))

            if self.use_cuda_graph:
                if cuda_graph_idx != -1:
                    self.features_buf[cuda_graph_idx][0].copy_(self.features[0], non_blocking=True)
                else:
                    self.features_buf[0].copy_(self.features[0], non_blocking=True)
                """
                if num_tokens_0 > 64:
                    padded_num_tokens_0 = pad_num_tokens(num_tokens_0)
                    self.features_buf[0][num_tokens_0:padded_num_tokens_0] = 0
                """
            #self.input.forward_minibatchs[0].print()
            # print([[hash(k[i].float().cpu().numpy().tobytes()) for i in self.input.forward_minibatchs[0].kv_indices] for k in self.model.cache.k_caches])
            # print(f"overlap: {overlap}, is_compute_bound: {is_compute_bound}")

            # self.model.flash_infer_attn_plan(self.input, self.bsz_tensors, self.num_tokens_tensors)

            """
            if self.use_cuda_graph:
                print("before replay features_buf", self.features_buf[0])
                print("features_buf addr", self.features_buf[0].data_ptr())
            else:
                print("before run features", self.features[0])
            """
            if cuda_graph_idx != -1 and self.use_cuda_graph:
                self.model.flash_infer_attn_plan(self.input[cuda_graph_idx], self.bsz_tensor_buf, self.num_tokens_tensor_buf,
                                            num_heads=self.model.config.num_attention_heads, head_dim_ckv=self.model.config.kv_lora_rank, 
                                                head_dim_kpe=self.model.config.qk_rope_head_dim, page_size=self.model.cache.page_size, causal=True,
                                                sm_scale=self.model.model.layers[0].self_attn.softmax_scale, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
                self.start_model_event.record(self.stream)
                page_idx, page_offset = self.model.cache.get_page_table(self.input[cuda_graph_idx].minibatch.position_ids, self.input[cuda_graph_idx].minibatch.q_indptr, self.input[cuda_graph_idx].minibatch.kv_indptr, self.input[cuda_graph_idx].minibatch.kv_indices, self.num_tokens_tensor_buf)
                if self.use_cuda_graph:
                    self.page_idx_buf[cuda_graph_idx][:num_tokens].copy_(page_idx[:num_tokens])
                    self.page_offset_buf[cuda_graph_idx][:num_tokens].copy_(page_offset[:num_tokens])
                    self.page_idx_buf[cuda_graph_idx][num_tokens:].fill_(self.model.cache.max_cache_len // self.model.cache.page_size - 1)
                    self.replay(cuda_graph_idx)
                    self.output = ForwardBatchOutput()
                    
                    self.output.top_ps.append(self.input[cuda_graph_idx].minibatch.top_ps)
                    self.output.temperatures.append(self.input[cuda_graph_idx].minibatch.temperatures)

                    self.output.logits.append(self.outputs_buf[cuda_graph_idx].logits[0][self.input[cuda_graph_idx].minibatch.logits_start].clone())
                else:
                    self.output = self.model(self.input[cuda_graph_idx], self.features, self.bsz_tensor_buf, self.num_tokens_tensor_buf, page_idx, page_offset)
                    self.output.logits[0] = self.output.logits[0][self.input[cuda_graph_idx].minibatch.logits_start]
                self.end_model_event.record(self.stream)
            else:
                self.model.flash_infer_attn_plan(self.input, self.bsz_tensor_buf, self.num_tokens_tensor_buf,
                                            num_heads=self.model.config.num_attention_heads, head_dim_ckv=self.model.config.kv_lora_rank, 
                                                head_dim_kpe=self.model.config.qk_rope_head_dim, page_size=self.model.cache.page_size, causal=True,
                                                sm_scale=self.model.model.layers[0].self_attn.softmax_scale, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
                self.start_model_event.record(self.stream)
                page_idx, page_offset = self.model.cache.get_page_table(self.input.minibatch.position_ids, self.input.minibatch.q_indptr, self.input.minibatch.kv_indptr, self.input.minibatch.kv_indices, self.num_tokens_tensor_buf)
                if self.use_cuda_graph:
                    self.page_idx_buf[:num_tokens].copy_(page_idx[:num_tokens])
                    self.page_offset_buf[:num_tokens].copy_(page_offset[:num_tokens])
                    self.page_idx_buf[num_tokens:].fill_(self.model.cache.max_cache_len // self.model.cache.page_size - 1) 
                    self.replay(cuda_graph_idx)
                    self.output = ForwardBatchOutput()

                    self.output.top_ps.append(self.input.minibatch.top_ps)
                    self.output.temperatures.append(self.input.minibatch.temperatures)

                    self.output.logits.append(self.outputs_buf.logits[0][self.input.minibatch.logits_start].clone())
                else:
                    self.output = self.model(self.input, self.features, self.bsz_tensor_buf, self.num_tokens_tensor_buf, page_idx, page_offset)
                    self.output.logits[0] = self.output.logits[0][self.input.minibatch.logits_start]
                    self.output.top_ps.append(self.input.minibatch.top_ps)
                    self.output.temperatures.append(self.input.minibatch.temperatures)

                self.end_model_event.record(self.stream)

        if not self.use_cuda_graph:
            self.output.num_batchs = self.input.batch_size
        else:
            self.output.num_batchs = self.input[cuda_graph_idx].batch_size


    def replay(self, cuda_graph_idx=-1):
        with torch.cuda.stream(self.stream):
            if cuda_graph_idx != -1:
                self.graphs[cuda_graph_idx].replay()
            else:
                self.graphs.replay()


    def sync(self, calc_time = True):
        self.stream.synchronize()
        if calc_time:
            self.model_time = self.start_model_event.elapsed_time(self.end_model_event)  # In ms
