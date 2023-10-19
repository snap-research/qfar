"""
Example script that shows a typical pipeline
"""
import numpy as np
from utils import run_detect, run_align, run_decode, run_decode_opencv

src_dir = 'data/distance'
run_detect(src_dir, ['--device', 'cpu'])
run_align(src_dir)
code_database = np.load(src_dir+'/database.npz')
run_decode(src_dir, code_database['database'], (int(code_database['code_size']),int(code_database['code_size'])))
# run_decode_opencv(src_dir) # Need to download wechat models and place them in wechat_models/