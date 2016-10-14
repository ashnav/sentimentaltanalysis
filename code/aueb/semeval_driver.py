from detect_sentiment import main as detect_sentiment

#read test data file
message_tests = {}
with open("../../data/test40", "r") as f:
   for line in f:
      line_split = line.split('\t')
      message_tests[line_split[2]] = line_split[0]

#run classifier
results = detect_sentiment(message_tests)

#write results
with open("../../data/aueb_original_results", "w") as results_out:
   for tweet_id, polarity in results:
      results_out.write(tweet_id +"\t" + polarity + "\n")

#author: Stephanie Durand   
