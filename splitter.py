from main import readFile
import random
import numpy as np
import os

os.mkdir('fold')
u, i, u_i_r = readFile("ratings.csv", type="train", type_return="array")

size_pop = len(u_i_r)
size_pop10 = int(0.1 * len(u_i_r))

for f in range(5):
    r = random.sample(range(size_pop), size_pop10)
    r = np.sort(r)

    cp = 0
    cs10 = 0
    with open(f"folds/ratings-{f}.csv", 'w') as fp:
        with open(f"folds/targets-{f}.csv", 'w') as fs10:
            fp.write("bl\n")
            fs10.write("bl\n")
            for i in range(size_pop):
                if cs10 < size_pop10:
                    if r[cs10] == i:
                        fs10.write("u" + str(u_i_r[i][0]).zfill(7) + ":" + "i" + str(u_i_r[i][1]).zfill(7) + "," + str(
                            u_i_r[i][2]) + ",1421526527" + "\n")
                        cs10 += 1
                    else:
                        fp.write("u" + str(u_i_r[i][0]).zfill(7) + ":" + "i" + str(u_i_r[i][1]).zfill(7) + "," + str(
                            u_i_r[i][2]) + ",1421526527" + "\n")

                        cp += 1
                else:
                    fp.write("u" + str(u_i_r[i][0]).zfill(7) + ":" + "i" + str(u_i_r[i][1]).zfill(7) + "," + str(
                        u_i_r[i][2]) + ",1421526527" + "\n")

                    cp += 1
