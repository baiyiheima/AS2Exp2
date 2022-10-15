def read_data(filename):
  with open(filename, 'r',encoding="utf8") as datafile:
    res = []
    count=0
    for line in datafile:
      count=count+1
      if count==1:
        continue
      #len(line) = 3
      line = line.strip().split('\t')
      str = line[1].strip().split(' ')
      if len(str)>500:
        print(line[1])
        #continue;
      res.append([line[0].lower(), line[1].lower(), float(line[2])])
  return res

if __name__ == '__main__':
    trainFilename = "D:/2022/ExpTwo/exp2/data/WikiQA/wikic_train.tsv"
    x = read_data(trainFilename)
    print(x)