import glob
import os
files = glob.glob("*.cl.ptx")

for ptx in files:
	inst = "ptxas {} -o {}.elf --gpu-name sm_61".format(ptx, ptx[:-4])
	print(inst)
	os.system(inst)