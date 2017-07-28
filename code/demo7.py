import argparse
parser = argparse.ArgumentParser(prog='PROG',description='''this
 description was indented wierd
but that is okey''',
epilog='''
likewise for this epilog whose whitespace will be
cleaned up and whose words will be wrapped
across a couple lines''')
parser.print_help()
