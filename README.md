# Financial-DL Blog

## Motivation
The stock market has been a longstanding icon of economic opportunity in American history. It has been notoriously difficult, if not impossible, to predict and profit from. Regardless of economic prosperity or decline, one question that’s arisen in the past few decades is the value of human insight versus the raw power of computers in managing portfolios. In this project, we analyze the performance of human investors using basic heuristics to buy and sell stocks against the performance of deep learning agents. We try three main approaches: human heuristic, LSTM + human heuristic (hybrid), and reinforcement learning. 



Human Heuristic
The most naive human heuristic could be just picking a random stock from a top company and investing in it. In such an approach we are not certain that we will make a profit or not. Like here we picked some random stocks to invest in and plotted the profit.
Here we can see we made a negative profit.


A better heuristic would be to take floating average of the stock price for the various companies. Check their current stock price and if the price lower than the company’s floating average and the market is favourable i.e. the market trend is positive. We invest for that company. Applying this heuristic the result we got are here. 





## Data and preprocessing

For our project we used the CRSP dataset for historical daily stock market data for 22 of the largest companies in the financial sector from 2009 to 2018. We queries the database for Price, Volume, Bid and Ask for our 22 selected tickers. 

22 stocks used in our project:
‘TROW’, ‘CMA’, ‘BEN’, ‘WFC’, ‘JPM’, ‘BK’, ‘NTRS’, ‘AXP’, ‘BAC’, ‘USB’, ‘RJF’, ‘C’, ‘STT’, ‘SCHW’, ‘COF’, ‘IVZ’,‘ETFC’,‘AMG’,‘GS’,‘BLK’,‘AMP’,‘DFS’

We cleaned and preprocessed our data by removing entries where the values was null for any of the fields and made sure that we had 10 years of data for each stock we considered. We tried min max normalization for our data as well but log normalizing the values yielded better performance in our LSTM so we decided to conduct all the experiments with log normalized data. This may have been due to the extreme variation in the data over the 10 year span. We use the first 9 years as training set (2009-2017), and the last 1 year as testing set (2018). 

Sample data for Goldman Sachs



## RL Environment

The Environment setup looks the past 30 days and to maximize the net worth of the next day. In order to make discrete action with the environment every model need to communicate two things per stock. First whether to buy sell hold the stock. We implement this by allowing the first set of logits three options if it outputs <1 buy, <2 sell, and <3 hold. Secondly we needed to determine a way of deciding the amount of potential stock to buy per an action. Based upon that using percentage total buy/sell didn’t work we decided on having the network produces discrete output between the ranges of [0,255] where 255 is the maximum shares of a stock that could be bought at one action set. 
	
## Methods
-	Human heuristic
	One of the most intuitive investment strategies in stock market is “Buy low, sell high”. Buying low means purchasing shares when stocks have hit bottom price. Conversely, selling high means selling shares when stocks have hit their peak. We implemented an adjusted “buy low, sell high” strategy as our human heuristic. An agent has an initial balance of $10,000. At time T, the agent observes the current stock prices (closing prices), as well as the historical prices of the past 30 days. A stock is considered at “low” price if the current price is lower than the average price over the past 30 days. Hence, the agent’s action is to buy the stock which has the lowest price (the most negative deviation from the average) with all the money in his balance. Meanwhile, the agent will sell all the shares currently held (arguably, these stocks are not at their lowest price on that day). We add a constraint on the number of buy and sell shares to simplify the comparison of different models. For each stock, buy and sell shares will be min(current balance, total value of 255 shares), which ensures the agent’s action and reward are within a feasible space. The performance of our human heuristic is evaluated as the daily returns using this strategy. To compare human heuristic baseline with other models, we compute the daily returns for the testing period (the last year). 

-	RNN (LSTM) + Human Heuristic
	For our RNN model we decided to use a simple LSTM that took in the time series data and outputted a prediction trend for D days. Initially we obtained pretty poor results by trying to predict a 30 day trend but as we decreased D our results were a lot better. Finally, we decided to predict just a single day as input to our human heuristic. 

Architecture of RNN model:


For each company we trained and the above model to give a single day point prediction given 30 days input. We tried batch sizes of 30, 50, 100  and 150, epochs in the range of 50, 100, and 150 and learning rates of .1, .01, and .001. We saw the best results with a batch size of 50 and 100 epochs trained with learning rate .001.  The 22 trained models loaded into our RL environment combined with our human heuristic allowed our agent to operate by getting a prediction and feeding it into the heuristic and selecting an action accordingly.

Sample predictions graphs on testing data (2018)



- 	RL
	Out of all the models above, the RL is the closest to represent what a human interprets when choosing stocks. This is because human chooses stock usually not only on the current market by also considers the current financial position, e.g. the portfolio. The input of the RL is broken down into two parts frames and portfolio. Firstly, the frames is simply same the input for rnn and the human heuristic is price, volume, bid and ask for the last 30 days. The difference is the portfolio also inputted into the model. The portfolio is broken down into balance (current cash), net worth (balance+value of all currently owned stocks), current shares held (list of the amount of currently held shares of a stock for each company), cost basis (what the average bought price for that stock currently owned for each company), total shares sold (the amount of that stock sold per company), total shares value which is the amount of money made by a certain stock. Both inputs were fed into a Proximal Policy Optimization (PPO) and trained with a simple reward function maximizing the net worth of it’s profilio.   

## Results


After running the different methods in the RL environment described previously we observed that the rnn and rl methods outperformed the human heuristic. The random baseline generally follows the market trend and thus is not expected to yield substantive results. In general after running multiple times we see that our rnn and rl models outperformed the human heuristic as seen in one such run above. One possible explanation for this is the naiveness of our human heuristic. In reality, humans factor much more information and base their decisions of more than just an average of competing stocks. Also our human heuristic is more susceptible to short term dips or repetitive patterns that may be caught by the deep learning models. One further consideration is that one year is not really a significant amount of time to test our models. Even within this period there is overlap between which models are outperforming the others. Extending the time frame might show that the models do not conclusively outperform one another. More testing over periods of economic prosperity as well as downturn would provide a more rigorous answer to which method is the best of the three.
 


## Future Scope
-	Stochastic method
	One pitfall of using the RNN is that given an input time series sequence the model would always predict the same point prediction. This sort of deterministic model tries to minimize the loss in whichever price it predicts. For example, if the model thinks than price will be either $100 or $200 it may choose to predict $150 to minimize it’s loss. In contrast, a stochastic model outputs the most likely value directly.

-	Use actual investors for human heuristic
	Currently, we use a simplified heuristic rule “Buy low, sell high” to mimic investors’ decisions. While it looks like a good strategy on paper, it essentially relies on figuring out the best time to buy and sell a stock. To time the stock market is extremely hard due to its unpredictable nature. Another downside of this strategy is that its return is very sensitive to the market trend. For example, an investor would be better-off using the “buy low” strategy during a bear market, where stock prices go down and investors tend to sell off shares as fear takes over. 
To overcome these limitations and obtain more realistic human investor performance as baseline for model comparison, we would invite investors to conduct experiments. In the lab experiments, we will sample several stocks from the 22 stocks in our data, and show each participant the actual prices of the stocks in 2018 (testing period). Participants will have the same $10,000 (as digital numbers in their computer account) as initial balance. We will ask participants to carefully observe and analyze the stock prices, and allow them to buy or sell any stocks with any proportion of their current balance on any day during 2018. We will take records of their investments and calculate final returns of each participant. Conducting experiments with actual investors also provides us with the rationales behind their decision making, which would be valuable for improving our RL algorithms. 

-	Biases and Abnormality of Data
Lastly the current model has been operated on with somewhat basis construction of the data. Firstly, starting with the premise of choosing the current top financial stocks creates a foundation with basis since if these are considered top currently they must have been performing comparatively well in the past decade in order to currently be known as one of the  top financial stocks. Next is when dealing with missing data in our model. Currently how we deal with some features abstinence in a datapoint is too skip it and consider the next datapoint as the current. This means that the space in between actions is not consistent and thus impacts our model. Finally, the testing set was just shy of one year in the future we should expand both the testing set and training set to prevent abnormalities that happen in short periods of time.    

## References

https://wrds-www.wharton.upenn.edu/pages/support/data-overview/wrds-overview-crsp-us-stock-database/

https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e

https://towardsdatascience.com/machine-learning-techniques-applied-to-stock-price-prediction-6c1994da8001

https://medium.com/@TalPerry/i-went-to-the-german-alps-and-applied-reinforcement-learning-to-financial-portfolio-optimization-b621c18a69d5

Cartea, A., Jaimungal, S. and Ricci, J., 2014. Buy low, sell high: A high frequency trading perspective. SIAM Journal on Financial Mathematics, 5(1), pp.415-444.
 
