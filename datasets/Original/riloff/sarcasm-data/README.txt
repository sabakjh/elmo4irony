This dataset contains sarcasm annotations for 3,000 tweets from
Twitter. Due to Twitter's data sharing restrictions, we are not
permitted to share the tweets, but we are permitted to share the tweet
ids so you can download the same tweets directly from Twitter. 
The tweet ids and our human judgements can be found in the file:
sarcasm-annos-emnlp13.tsv

Format: tab separated. Each line contains a tweet id followed by the
label (SARCASM or NOT_SARCASM), separated by a tab. For example:

240465764967141376 NOT_SARCASM

This data set was used for research described in the following paper:

Riloff, E., Qadir, A., Surve, P., De Silva, L., Gilbert, N., and
Huang, R. (2013) "Sarcasm as Contrast between a Positive Sentiment and
Negative Situation", Proceedings of the 2013 Conference on Empirical
Methods in Natural Language Processing (EMNLP 2013).

URL: http://www.cs.utah.edu/~asheq/publications/sarcasm-qadir-EMNLP-2013.pdf

If you use this data in your own research, please cite this paper:

@inproceedings{riloff-emnlp13,
    author = "Riloff, E. and Qadir, A. and Surve, P. and De Silva, L. and Gilbert, N. and Huang, R.",
    title = "{Sarcasm as Contrast between a Positive Sentiment and Negative Situation}", 
    booktitle = "{Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)}",
    year = 2013
}

=============================================================================
ADDITIONAL DETAILS ABOUT THE SARCASM ANNOTATIONS

The tweets were collected using Twitter's Streaming API v1.0
(https://dev.twitter.com/docs/streaming-apis) in the last week of
August 2012. Only English tweets were retained using an open source
language detection tool, "ldig"
(http://shuyo.wordpress.com/2012/02/21/language-detection-for-twitter-with-99-1-accuracy/).
Unfortunately, it is possible that some of the tweets may no longer be
available from Twitter. 

We obtained good pairwise inter-annotator agreement scores among three
human annotators based on Cohen's kappa (kappa >= .80) for these
sarcasm judgements. However, despite our best efforts, sarcasm is a
highly subjective phenomenon and human annotation is not error
free. So there may be some judgements that you would disagree with.
Overall, however, we hope that you will find these sarcasm judgements
to be of high quality!

If you have further questions, please contact:

Ashequl Qadir 
asheq@cs.utah.edu 

Ellen Riloff
riloff@cs.utah.edu



