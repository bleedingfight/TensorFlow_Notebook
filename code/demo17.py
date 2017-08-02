import math
import argparse
def perfect_square(string):
    value = int(string)
    sqrt = math.sqrt(value)
    if sqrt!=int(sqrt):
        msg='%r is not a perfect square'%string
        raise argparse.ArgumentTypeError(msg)
    return value
parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('foo',type=perfect_square)
print(parser.parse_args(['9']))
print(parser.parse_args(['7']))

