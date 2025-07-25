"""
 This file demonstrates the example usage of disaggregated prefilling
 We will launch 2 vllm instances (NPU 0,1 for prefill and NPU 2,3 for decode),
 and then transfer the KV cache between them.
 prompy_device_ips denotes device ip of NPU 0,1
 decode_device_ips denotes device ip of NPU 2,3
 The device ips of all NPUs in current server can be found through
 examples/disaggregated_prefill/find_device_ips.py
 """
import multiprocessing as mp
import os
import time
from multiprocessing import Event, Process

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

kv_connector_extra_config = {
    "prefill_device_ips": ["1.2.3.1", "1.2.3.2"],
    "decode_device_ips": ["1.2.3.9", "1.2.3.10"],
    "llmdatadist_comm_port": 26000,
}


def clean_up():
    import gc

    import torch
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment, destroy_model_parallel)
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()


def run_prefill(prefill_done, process_close):
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    prompts = [
        "Hello, how are you today?", "Hi, what is your name?",
        "Tell me a very long story.", "what is your favourite book?"
    ]
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"AscendSimpleConnector","kv_buffer_device":"npu","kv_role":"kv_producer", "kv_parallel_size":2}'
    )
    global kv_connector_extra_config
    ktc.kv_connector_extra_config = kv_connector_extra_config
    llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
              kv_transfer_config=ktc,
              max_model_len=2000,
              gpu_memory_utilization=0.8,
              tensor_parallel_size=2)

    llm.generate(prompts, sampling_params)
    print("Prefill node is finished.")
    prefill_done.set()

    # To keep the prefill node running in case the decode node is not done;
    # otherwise, the script might exit prematurely, causing incomplete decoding.
    try:
        while not process_close.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("Script stopped by user.")
    finally:
        print("Cleanup prefill resources")
        del llm
        clean_up()


def run_decode(prefill_done):
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "2,3"

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    prompts = [
        "Hello, how are you today?",
        "Hi, what is your name?",
    ]
    sampling_params = SamplingParams(temperature=0, top_p=0.95)

    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"AscendSimpleConnector","kv_buffer_device":"npu","kv_role":"kv_consumer","kv_parallel_size":2}'
    )
    global kv_connector_extra_config
    ktc.kv_connector_extra_config = kv_connector_extra_config
    llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
              kv_transfer_config=ktc,
              max_model_len=2000,
              gpu_memory_utilization=0.8,
              tensor_parallel_size=2)

    # Wait for the producer to start the consumer
    print("Waiting for prefill node to finish...")
    prefill_done.wait()

    # At this point when the prefill_done is set, the kv-cache should have been
    # transferred to this decode node, so we can start decoding.
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    del llm
    clean_up()


if __name__ == "__main__":
    mp.get_context('spawn')

    prefill_done = Event()
    process_close = Event()
    prefill_process = Process(target=run_prefill,
                              args=(
                                  prefill_done,
                                  process_close,
                              ))
    decode_process = Process(target=run_decode, args=(prefill_done, ))

    # Start prefill node
    prefill_process.start()

    # Start decode node
    decode_process.start()

    # Terminate the prefill node when decode is finished
    decode_process.join()

    # Terminate prefill process
    process_close.set()
    prefill_process.join()
    prefill_process.terminate()
    print("All process done!")
