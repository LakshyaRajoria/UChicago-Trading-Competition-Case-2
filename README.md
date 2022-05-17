# UChicago-Trading-Competition-Case-2
Calculating Fair on Options and Market Making

For this case our strategy first dealt with constructing a model to find implied vol which we could then input back into the Black Scholes model and get a fair for the options. The way we do this is we use Garch, because it models periods of volatility swings really well. This was especially important for this case as there were three events, a global pandemic scare, market uncertainty and a speculative bubble that would affect the market. 

Once we had that, we averaged it with the IV of the asset when it was being traded on the market because that’s just how typically we solve for the BS fair. Now we had our volatility model set up. The other thing we did with volatility is implement code that took into account vol skew. Each of the three events mentioned before has an impact on vol, such as in pandemics people want more downside protection so puts should become more expensive. In this case we want to position ourselves to be more long puts so we adjust our fair accordingly. To actually calculate the vol skew we used the cvol calculation, taking the 25-delta call and put and finding the difference in their IVs. Based on the result we updated our vol estimates, which directly impacts our fair value of options. 

One way to improve this bot would be to not run the GARCH vol prediction model every time we get a price update from the exchange. This made our fair prediction take too long to calculate in comparison with the speed of messages we were getting from the exchange. To ameliorate this, we could have simply used the last 10 data points as an input to GARCH vol function. Having said that, we did not test it out and so there is the possibility that the vol prediction isn’t as accurate as we would want it to be given that there isn’t as much data. 

In this case we placed 14 from the 40 teams we were competing against. 

