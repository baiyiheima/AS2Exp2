
def read_data(filename):
  with open(filename, 'r') as datafile:
      qas_res = []
      label_res = []
      count = 0
      for line in datafile:
          count = count + 1
          if (count == 1):
              continue
          line = line.strip().split('\t')
          lines = []
          length = len(line)
          if (length < 3):
              lines.append(line[0])
              lines.append("question")
              lines.append(line[1])
          else:
              lines.append(line[0].lower())
              lines.append(line[1].lower())
              lines.append(line[2])

          qas_res.append("[CLS] "+lines[0]+" [SEP] "+lines[1]+" [SEP]")
          label_res.append(float(lines[2]))
  return qas_res, label_res

def read_data_for_predict(filename):
  with open(filename, 'r') as datafile:
      res = []
      count = 0
      for line in datafile:
          count = count + 1
          if (count == 1):
              continue
          line = line.strip().split('\t')
          lines = []
          length = len(line)
          if (length < 3):
              lines.append(line[0])
              lines.append("question")
              lines.append(line[1])
          else:
              lines.append(line[0])
              lines.append(line[1])
              lines.append(line[2])

          res.append([lines[0], lines[1], float(lines[2])])
  return res