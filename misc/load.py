import torch
import time
import sys
import argparse

# --- CONFIGURATION ---
SIZE = 50_016
SLEEP_TIME = 10
# ---------------------

def main(args):
    '''
    Main function

    python load.py --intensity large
    python load.py --size 5016 --sleep_time 10
    python load.py --intensity large --gpu 0
    python load.py --intensity large --gpu 0 2 3
    '''

    # variables
    size = args.size
    sleep_time = args.sleep_time
    if args.intensity:
        print(f"Proceeding with intensity '{args.intensity}'...")
        if args.intensity == "small" or args.intensity == "s":
            size = 100
            sleep_time = 100
        elif args.intensity == "medium" or args.intensity == "m":
            size = 25_016
            sleep_time = 10
        elif args.intensity == "large" or args.intensity == "l":
            size = 50016
            sleep_time = 10
        elif args.intensity == "xlarge" or args.intensity == "xl":
            size = 90_016
            sleep_time = 5
    else:
        print("Proceeding with custom size and sleep time settings.")
    print(f"Using size={size} and sleep_time={sleep_time}ms")
    
    # Device
    if not torch.cuda.is_available():
        print("CUDA (GPU) is not available. Exiting.")
        sys.exit(1)
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found. Exiting.")
        sys.exit(1)
    
    # Parse GPU selection
    if args.gpu is not None:
        for gpu_id in args.gpu:
            if gpu_id < 0 or gpu_id >= num_gpus:
                print(f"Error: GPU {gpu_id} is not available. Valid range: 0-{num_gpus-1}")
                sys.exit(1)
        devices = [torch.device(f"cuda:{i}") for i in args.gpu]
    else:
        devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
    
    if len(devices) == 1:
        print(f"Using GPU: {torch.cuda.get_device_name(devices[0])}")
    else:
        print(f"Using {len(devices)} GPUs:")
        for device in devices:
            print(f"  - {torch.cuda.get_device_name(device)}")
    
    print(f"Creating tensors of size {size}x{size}...")

    try:
        # Create tensors on each GPU
        tensors = {}
        for device in devices:
            tensors[device] = {
                'a': torch.randn(size, size, device=device),
                'b': torch.randn(size, size, device=device)
            }
        print("Starting GPU calculation ...")

        while True:
            # Matrix Multiplication on each GPU
            for device in devices:
                c = torch.matmul(tensors[device]['a'], tensors[device]['b'])
            
            # This sleep controls the duty cycle.
            # The GPU is busy for X ms (the matmul) and idle for Y ms (the sleep).
            # GPU Load % ≈ (X / (X + Y))
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Stopping GPU load.")
    except RuntimeError as e:
        print(f"Error (likely Out of Memory): {e}")
        print("Try decreasing the 'size' variable in the script.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", type=int, default=SIZE)                       # Increase this value to make the calculation heavier
    parser.add_argument("-t", "--sleep_time", type=int, default=SLEEP_TIME)           # Decrease this to run the loop more often (higher GPU load) (in ms)
    parser.add_argument("-i", "--intensity", type=str, default=None)                  # Run Intensity: [small, medium, large, xlarge]
    parser.add_argument("-g", "--gpu", type=int, nargs='+', default=None, help="Space-separated GPU IDs (e.g., 0 or 0 2 3). If not specified, uses all GPUs.")
    args = parser.parse_args()
    main(args)

