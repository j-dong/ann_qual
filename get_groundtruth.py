import argparse
import struct

def read_int(x: bytes) -> int:
    return struct.unpack('<I', x)[0]

parser = argparse.ArgumentParser()
parser.add_argument('i', type=int)
args = parser.parse_args()

with open('G:/vectors/siftsmall/siftsmall_groundtruth.ivecs', 'rb') as f:
    dim = read_int(f.read(4))
    f.seek((1 + dim) * 4 * args.i + 4)
    for i in range(dim):
        print(read_int(f.read(4)))
