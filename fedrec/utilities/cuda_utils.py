#------------------------------------------------
""" 
importing necessary (libraries) 
torch ---> PyTorch is a Python package that provides two high-level features:
Tensor computation (like NumPy) with strong GPU acceleration
Deep neural networks built on a tape-based autograd system

socket ---> Python package which allows creation of simple servers and clients for communication with sockets. Supports both Python2 and Python3.
logging ---> This module is intended to provide a standard error logging mechanism in Python as per PEP 282.

"""
import torch
import socket
import logging
"""
function name ---> map_to_cuda 
if the input to the function is of type "List" or "Tuple" it will return [map_to_cuda(arg, device, **kwargs) for arg in args]
if the input to the function is of type "Dictionary" it will return {k: map_to_cuda(v, device, **kwargs) for k, v in args.items()}
if the input to the function is of type "torch.Tensor" it will return args.cuda(device, **kwargs)
if there is unmatched of all it will raise TypeError "unsupported type for cuda migration"
"""

def map_to_cuda(args, device=None, **kwargs):
    if isinstance(args, (list, tuple)):
        return [map_to_cuda(arg, device, **kwargs) for arg in args]
    elif isinstance(args, dict):
        return {k: map_to_cuda(v, device, **kwargs) for k, v in args.items()}
    elif isinstance(args, torch.Tensor):
        return args.cuda(device, **kwargs)
    else:
        raise TypeError("unsupported type for cuda migration")

"""
function name--> map_to_list
input--> model_params
It will convert the input (model_params) to list and return it
"""
def map_to_list(model_params):
    for k in model_params.keys():
        model_params[k] = model_params[k].detach().cpu().numpy().tolist()
    return model_params

"""
function name--> mapping_processes_to_gpus
inputs--> gpu_config, process_id, worker_number
if the gpu_config (input) is None then the device will be cpu and it will return the device
else it will return the total description of GPUs present 
informations it will print---->
        logging.info("Process: %d" % (process_id))
        logging.info("host: %s" % (gpu_util_map[process_id][0]))  
        logging.info("gethostname: %s" % (socket.gethostname()))  
        logging.info("gpu: %d" % (gpu_util_map[process_id][1]))
        
Now as GPU or GPUs are present it will be returned as device with their logging info 
"""
def mapping_processes_to_gpus(gpu_config, process_id, worker_number):
    if gpu_config == None:
        device = torch.device("cpu")
        logging.info(device)
        # return gpu_util_map[process_id][1]
        return device
    else:
        logging.info(gpu_config)
        gpu_util_map = {}
        i = 0
        for host, gpus_util_map_host in gpu_config.items():
            for gpu_j, num_process_on_gpu in enumerate(gpus_util_map_host):
                for _ in range(num_process_on_gpu):
                    gpu_util_map[i] = (host, gpu_j)
                    i += 1
        logging.info("Process: %d" % (process_id))
        logging.info("host: %s" % (gpu_util_map[process_id][0]))
        logging.info("gethostname: %s" % (socket.gethostname()))
        logging.info("gpu: %d" % (gpu_util_map[process_id][1]))
        assert i == worker_number

        device = torch.device(
            "cuda:" + str(gpu_util_map[process_id][1])
            if torch.cuda.is_available() else "cpu")
        logging.info(device)
        # return gpu_util_map[process_id][1]
        return device
