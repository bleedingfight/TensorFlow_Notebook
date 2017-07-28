import argparse
parser = argparse.ArgumentParser(prog='PROG',
formatter_class=argparse.RawDescriptionHelpFormatter,
description = textwrap.dedent('''\
Plase do not mess up this text!
------------------------------
I have idented it
exactly the way
I want it
'''))
parser.print_help()
