from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import qrcode

def make_database_from_image(src_dir, num_codes=10000, version=1, msg_length=25, seed=20220718):
    # Read code image
    img = cv2.imread(str(Path(src_dir).joinpath('code0_raw.png')), cv2.IMREAD_GRAYSCALE)
    code0 = (img>0).astype(float).flatten()
    # Create database
    size = 17 + version * 4
    code_database = np.zeros((num_codes, size*size))
    alpha_num = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:'
    rng = np.random.default_rng(seed)
    for i in tqdm(range(num_codes)):
        data_ind = rng.integers(len(alpha_num), size=msg_length)
        data = ''.join([alpha_num[i] for i in data_ind])
        qr = qrcode.QRCode(
                version=version,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=1,
                border=0
        )
        qr.add_data(data)
        qr.make(fit=False)
        qr_img = qr.make_image(fill_color='black', back_color='white')
        qr_arr = np.asarray(qr_img).astype(float)
        code_database[i,:] = qr_arr.flatten()
    # Inject the code into the database
    code_database[0,:] = code0
    # Save the database
    np.savez_compressed(Path(src_dir).joinpath('database.npz'), database=code_database, code_size=size)