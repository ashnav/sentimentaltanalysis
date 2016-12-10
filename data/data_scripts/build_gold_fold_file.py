from code.twitter_data_reader import read_file
for foldNum in range(0,10):
    results = "data/weight_change_results/combined_results_fold" + str(foldNum) + ".out"
    gold = "data/semeval_data.tsv"
    gold_fold ="scorers/gold_fold_weight"+str(foldNum)

    tweets = read_file(gold)
    ids = []
    with open(results, 'r') as readfile:
       for line in readfile:
          tweet_id = line.split("\t")[0]
          ids.append(tweet_id)

    with open(gold_fold, "w") as goldfile:
       for tweet_id in ids:
          for x in range(0, len(tweets)):
             if tweets[x]['ID'] == tweet_id:
                goldfile.write("{}\t{}\n".format(tweet_id, tweets[x]['SENTIMENT']))
                break
         


