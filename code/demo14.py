import argparse
parser = argparse.ArgumentParser(prog='Prog',allow_abbrev=False)
parser.add_argument('--foobar',action='store_true')
parser.add_argument('--fooley',action='store_true')
parser.parse_args(['--foon'])
