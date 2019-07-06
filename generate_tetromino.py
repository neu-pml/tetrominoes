import sys
import os
import pickle
from datasets import *
from utils import save_file

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

if __name__ == "__main__":

    DATA_PATH = sys.argv[1]
    if not os.path.isdir(DATA_PATH):
        os.mkdir(DATA_PATH)

    id_tetrominos_filename = os.path.join(DATA_PATH, "id_tetrominos")
    ood_tetrominos_filename = os.path.join(DATA_PATH, "ood_tetrominos")

    cs = checkerboard_pattern_5d(n_checker=1)
    id_tetrominos = TETROMINOS(sample="stratified_grid", train_ratio=0.5)
    ood_tetrominos = TETROMINOS(sample="grid", constraints=cs)

    save_file(id_tetrominos_filename, id_tetrominos)
    save_file(ood_tetrominos_filename, ood_tetrominos)

    if len(sys.argv) > 2:
        for tr in sys.argv[2:]:
            id_tetrominos_tr_filename = os.path.join(DATA_PATH, "id_tetrominos_tr=%s" % tr)
            id_tetrominos_tr = TETROMINOS(sample="stratified_grid", train_ratio=float(tr))
            save_file(id_tetrominos_tr_filename, id_tetrominos_tr)
