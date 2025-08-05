import os

class MemoryAwareWorkerManager:
    def __init__(self):
        self.total_memory = get_gpu_memory()
        self.used_memroy = get_used_memory()
        self.mem_margin = os.getenv("VBENCH_FILL_DEVICE_BUFFER_SIZE", 0.1)
        self.memory_estimate = {
            "activation":
            "reserved":
        }
    
    def reduce_workers(self):
        if len(self.workers) > 1:
            worker = self.workers.pop()
            worker.terminate()
    
    def add_worker(self):
        estimated_additional_memory = self.estimate_worker_memory()
        available_memory = self.get_available_memory()
        
        if estimated_additional_memory < available_memory:
            new_worker = self.create_worker()
            self.workers.append(new_worker)
