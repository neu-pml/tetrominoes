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

    id_tetrominoes_filename = os.path.join(DATA_PATH, "id_tetrominoes")
    ood_tetrominoes_filename = os.path.join(DATA_PATH, "ood_tetrominoes")

    id_tetrominoes = Tetrominoes(mode="id")
    ood_tetrominoes = Tetrominoes(mode="ood")

    save_file(id_tetrominoes_filename, id_tetrominoes)
    save_file(ood_tetrominoes_filename, ood_tetrominoes)

    if len(sys.argv) > 2:
        for tr in sys.argv[2:]:
            id_tetrominoes_tr_filename = os.path.join(DATA_PATH, "id_tetrominoes_tr=%s" % tr)
            id_tetrominoes_tr = Tetrominoes(train_ratio=float(tr))
            save_file(id_tetrominoes_tr_filename, id_tetrominoes_tr)
