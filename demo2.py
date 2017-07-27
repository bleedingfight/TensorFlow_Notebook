import argparser
parser = argparse.ArgumentParser(prog='Prog')
parser.add_argument('--foo',nargs='?',help='foo help')
parser.add_argument('bar',nargs='+',help='bar help')
parser.print_help()
