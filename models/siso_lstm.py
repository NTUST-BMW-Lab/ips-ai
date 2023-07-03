import argparse
import datetime
import numpy as np
import pandas as pd
import os
import multiprocessing

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
num_cpu = multiprocessing.cpu_count()
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=num_cpu,
    inter_op_parallelism_threads=num_cpu
)

def siso_lstm(
        dataset: str,
        preprocessor: str,
        batch_size: int,
        epochs: int,
        optimizer: str,
        dropout: float,
):
    pass