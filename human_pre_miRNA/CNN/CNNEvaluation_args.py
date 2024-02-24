import getopt
import sys

def process_argv():
  requireds = ["input", "model", "output"]

  try:
    longopts = [ opt + "=" for opt in requireds ]
    opts, args = getopt.getopt(sys.argv[1:], "", longopts)
  except getopt.GetoptError:
    print("Wrong usage!")
    usage()
    sys.exit(1)

  # parse the options
  r = { 'verbose': 1 }
  for op, value in opts:
    if op in ("--input"):
      r['input'] = value
    elif op in ("--model"):
      r['model'] = value
    elif op in ("--output"):
      r['output'] = value
    elif op in ("--verbose"):
      r['verbose'] = int(value)
    elif op in ("-h","--help"):
      usage()
      sys.exit()

  for required in requireds:
    if not required in r:
      print("Wrong usage!!")
      usage()
      sys.exit(1)

  return r

def usage():
  print("USAGE: python CNNEvaluation_chiquitto.py --test test.csv --model model.h5 --output directory")
