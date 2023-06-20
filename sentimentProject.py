from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

#tweet = "@Anamitra  😀  @ cricket 🔥 https://twitter.com/AnamitraO"
#tweet = 'he is neither a good boy nor a bad boy'
#tweet='The Kerala Story’ is centered around the alleged radicalization and conversion of young Hindu women to Islam in Kerala, after which they are forced to join ISIS. The film states that it’s a true story of three young girls from different parts of Kerala.'
tweet='Hindus should realize these movies are made to exploit their emotions and innocence '

# precprcess tweet
tweet_words = []

for word in tweet.split(' '):
    if word.startswith('@') and len(word) > 1:
        word = '@user'
    
    elif word.startswith('http'):
        word = "http"
    tweet_words.append(word)

tweet_proc = " ".join(tweet_words)

# load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative🥲: ', 'Neutral😒: ', 'Positive😁: ']

# sentiment analysis
encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
# output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
output = model(**encoded_tweet)

scores = output[0][0].detach().numpy()
#scores = softmax(scores)

for i in range(len(scores)):
    
    l = labels[i]
    s = scores[i]
    print(l,s)
    
    #for loop statements use below code and try to get result using matplotlib

# from matplotlib import pyplot as plt
# Negative=[i]
# Neutral=[i]
# Positive=[i]
# plt.hist(Negative)
# plt.hist(Neutral)
# plt.hist(Positive)
 
# # Function to show the plot
# plt.show()    


# In[6]:
 #run this code separate and put all the value in right section then only u get the right ans 

from matplotlib import pyplot as plt
Negative=[0.78357977]
Neutral=[1.2079024]
Positive=[ -2.133885]
plt.hist(Negative)
plt.hist(Neutral)
plt.hist(Positive)
 
# Function to show the plot
plt.show()


# In[ ]:





# In[ ]:




