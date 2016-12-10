from twitter_data_reader import read_file


tweets = read_file("../data/semeval_data.tsv")
num_train = int(len(tweets) * 0.6)
header_string = "@relation tweets_sentiment \n@attribute \'tweet\'\tstring\n@attribute \'sentiment\'\t{positive,neutral,negative}\n@data\n"
with open("semeval_60.arff", "w") as trainfile:
   trainfile.write(header_string)
   for x in range(0, num_train):
      data = "\'{tweet}\',{sentiment}\n".format(sentiment=tweets[x]['SENTIMENT'],tweet=tweets[x]['MESSAGE'].strip().replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"'))
      trainfile.write(data)

with open("semeval_40.arff", "w") as testfile:
   testfile.write(header_string)
   for x in range(num_train, len(tweets)):
      data = "\'{tweet}\',{sentiment}\n".format(sentiment=tweets[x]['SENTIMENT'],tweet=tweets[x]['MESSAGE'].strip().replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"'))
      testfile.write(data)      

