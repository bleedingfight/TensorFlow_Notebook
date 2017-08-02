import argparse
parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('-f','--foo')
parser.add_argument('bar')
print(parser.parse_args(['BAR']))
print(parser.parse_args(['BAR','--foo','FOO']))


