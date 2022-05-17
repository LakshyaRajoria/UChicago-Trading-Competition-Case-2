
'''
TODO:
- Laddering Quotes (Uniformed client)
- Dynamic parameters (make gui or use exec to add update fade severity and manually sell off some positions with market orders)
- Vectorize python or Re-Write in C? TP exchange so speed matters a bit
- This has the vol skew function integrated with the black scholes fair
'''

# Imports:

# For Exchange:
from dataclasses import astuple
from utc_bot import UTCBot, start_bot
import proto.utc_bot as pb
import betterproto

import asyncio
import random

# For Bot:
import py_vollib.black_scholes.greeks.analytical
import py_vollib.black_scholes.implied_volatility
from arch import arch_model

# General:
import numpy as np
import pandas as pd
import math



option_strikes = [90, 95, 100, 105, 110]


class goofygoober(UTCBot):
	'''
	-------------------------------------------------- General Exchange Functions --------------------------------------------------
	'''
	async def handle_round_started(self):
		'''
		Function: handle_round_started
		Purpose: Handles starting the round, setting variables, etc.
		Params: none
		'''
		# General Exchange Information:
		self.asset_names = ['UC']
		for strike in option_strikes:
			for flag in ["C", "P"]:
				self.asset_names.append(f"UC{strike}{flag}")
		self.current_day = 0
		self.historical_pnl = []

		# Position/Price Info:
		self.positions = {}
		self.current_mrkt_price = {}
		self.current_asset_book = {}
		for asset in self.asset_names:
			self.positions[asset] = 0
			self.current_mrkt_price[asset] = 0
			self.current_asset_book[asset] = 0

		# GARCHCalculation:
		self.Garch_day = 0
		self.underlying_price = 100
		self.rolling_GARCH = pd.DataFrame(columns = ['price', 'returns'])
		vals_to_append = np.asarray([100, 0.0001])
		self.rolling_GARCH.loc[0] = vals_to_append


		self.current_vega_weighted_IV = 0

		# Order Details:
		self.delta_order = None

		self.bid_order_ids = {}
		self.ask_order_ids = {}
		for asset in self.asset_names:
			self.bid_order_ids[asset] = None
			self.ask_order_ids[asset] = None


	async def handle_exchange_update(self, update: pb.FeedMessage):
		'''
		Function: handle_exchange_update
		Purpose: Handles updates from the exchange
		Params: update (raw update message from the exchange)
		'''
		kind, _ = betterproto.which_one_of(update, "msg")

		if kind == "pnl_msg":
			print('My PnL: ', update.pnl_msg.m2m_pnl)
			self.historical_pnl.append(update.pnl_msg.m2m_pnl)

		elif kind == "fill_msg":
			asset = update.fill_msg.asset
			if asset == 'UC':
				if update.fill_msg.order_id == self.delta_order and update.fill_msg.remaining_qty == 0:
					self.delta_order = None

			if update.fill_msg.order_side == pb.FillMessageSide.BUY: # we were filled on a bid
				self.positions[asset] += update.fill_msg.filled_qty
				if update.fill_msg.order_id == self.bid_order_ids[asset] and update.fill_msg.remaining_qty == 0:
					self.bid_order_ids[asset] = None
			else: # we were filled on a ask
				self.positions[asset] -= update.fill_msg.filled_qty
				if update.fill_msg.order_id == self.ask_order_ids[asset] and update.fill_msg.remaining_qty == 0:
					self.ask_order_ids[asset] = None


		elif kind == "market_snapshot_msg":
			for asset in self.asset_names:
				# Get full book:
				asset_book = update.market_snapshot_msg.books[asset]
				self.current_asset_book[asset] = asset_book


				# Get swmid:
				try:
					best_bid = asset_book.bids[0]
					best_ask = asset_book.asks[0]
				except:
					# print(asset, ' 1-ERROR: CANNOT GET ASKS[0]')
					continue
				total_size = (best_bid.qty + best_ask.qty)
				if total_size != 0:
					swmid = ((float(best_bid.px)*best_bid.qty) + (float(best_ask.px) * best_ask.qty))/ (best_bid.qty + best_ask.qty)
				else:
					swmid = ((float(best_bid.px)*best_bid.qty) + (float(best_ask.px) * best_ask.qty))/2
				self.current_mrkt_price[asset] = (float(best_bid.px) + float(best_ask.px))/2# swmid

				# Garch Prices:
				if asset == 'UC':
					self.underlying_price = swmid
			await self.hedge_delta()

			# GARCH Setup:
			self.Garch_day += 1
			day = self.Garch_day
			p = self.underlying_price
			p_1 = self.rolling_GARCH['price'][day-1]
			r = ((p / p_1) -1)*100
			if r==0:
				r = 0.0001
			vals_to_append  = np.asarray([p, r])
			self.rolling_GARCH.loc[day] = vals_to_append

			await self.place_quotes()

		elif (kind == "generic_msg" and update.generic_msg.event_type == pb.GenericMessageType.MESSAGE):
			print(float(update.generic_msg.message))

		elif kind == "trade_msg":
			pass
			# print(update.trade_msg)

	'''
	-------------------------------------------------- Market Making Functions --------------------------------------------------
	'''
	async def place_quotes (self):
		'''
		Function: place_quotes
		Purpose: Calculates optimal quote based on BS fair, fade, size, laddering, etc.
		Params: None
		'''
		print(self.positions)
		for strike in option_strikes:
			for flag in ["C", "P"]:
				S = self.current_mrkt_price["UC"]
				K = strike
				t = (26 - self.current_day) / 252
				r = 0
				BSM_price = self.black_scholes_price(S, K, t, r, flag)
				if BSM_price == None:
					continue

				asset_book = self.current_asset_book[f"UC{strike}{flag}"]
				best_bid_error = False
				best_ask_error = False

				try:
					best_bid = asset_book.bids[0]
				except:
					best_bid_error = True
				try:
					best_ask = asset_book.asks[0]
				except:
					best_ask_error = True

				if best_ask_error and not best_bid_error:
					best_ask = best_bid
					best_ask.px += 0.1
				if best_bid_error and not best_ask_error:
					best_bid = best_ask
					best_bid.px -= 0.1
				if best_bid_error and best_ask_error:
					continue

				market_spread = float(best_ask.px) - float(best_bid.px)

				if market_spread < 0:
					#iTs FrEe ReAl EsTaTe! (fr tho its free EV so we should execute on it asap)
					# to do this we place an ask equal to their bid and a bid equal to their ask, size it to 1/2 bc we wanna get filled fast
					await self.place_bid_update(f"UC{strike}{flag}", int(float(best_ask.qty))/2, float(best_ask.px), None)
					await self.place_ask_update(f"UC{strike}{flag}", int(float(best_bid.qty))/2, float(best_bid.px), None)
					continue # We wanna wait for this to fill before we place a real quote

				# Determine current holdings and use it for linear fading
				size = self.positions[f"UC{strike}{flag}"]
				fade_severity = 3 # adjust on the fly if we think we are underinformed (move up) / overinformed (move down)
				fade_amt = (-0.1)*((size/fade_severity)**1.3)
				BSM_price += fade_amt # For some reason fading is tanking pnl need to figure out why

				# Determine slack params for wideness
				slack_size = 2 #Adjust on the fly (adjust down if we need to get filled more/move more size)
				slack = market_spread/slack_size
				ideal_bid = BSM_price - slack
				ideal_ask = BSM_price + slack
				if ideal_bid <= float(best_bid.px):
					ideal_bid = float(best_bid.px)+0.1
				if ideal_ask >= float(best_ask.px):
					ideal_ask = float(best_ask.px)-0.1
				current_spread = ideal_ask - ideal_bid

				if current_spread <=0:
					ideal_bid = float(best_bid.px)
					ideal_ask = float(best_ask.px)

				ideal_bid = round(ideal_bid,1)
				ideal_ask = round(ideal_ask,1)

				lot_size = 3 #should be changed on the fly but this is a safe bet given a max pos of 15 on any contract
				imposed_limit = 35
				if (size + lot_size) > imposed_limit:
					ideal_bid = None
					if self.bid_order_ids[f"UC{strike}{flag}"] != None:
						await self.cancel_order(self.bid_order_ids[f"UC{strike}{flag}"])

				if (size - lot_size) < -imposed_limit:
					ideal_ask = None
					if self.bid_order_ids[f"UC{strike}{flag}"] != None:
						await self.cancel_order(self.bid_order_ids[f"UC{strike}{flag}"])

				await self.handle_quote_placement(f"UC{strike}{flag}", ideal_bid, ideal_ask, lot_size)





	async def place_bid_update(self, asset, size, limit, order_id):
		'''
		Function: place_bid_update
		Purpose: Updates/places bid order
		Params: Asset (i.e. UC100C), size of order, limit price of order, orderId to update
		'''
		if limit == None:
			return
		elif order_id == None: # place single-use order
			bid_response = await self.place_order(asset, pb.OrderSpecType.LIMIT, pb.OrderSpecSide.BID, int(size), limit)
			self.bid_order_ids[asset] = bid_response.order_id
			return
		else:
			await self.modify_order(order_id, asset, pb.OrderSpecType.LIMIT, pb.OrderSpecSide.BID, int(size), limit)


	async def place_ask_update(self, asset, size, limit, order_id):
		'''
		Function: place_ask_update
		Purpose: Updates/places ask order
		Params: Asset (i.e. UC100C), size of order, limit price of order, orderId to update
		'''
		if limit == None:
			return
		elif order_id == None: # place single-use order
			ask_response = await self.place_order(asset, pb.OrderSpecType.LIMIT, pb.OrderSpecSide.ASK, int(size), limit)
			self.ask_order_ids[asset] = ask_response.order_id
		else:
			await self.modify_order(order_id, asset, pb.OrderSpecType.LIMIT, pb.OrderSpecSide.ASK, int(size), limit)


	async def handle_quote_placement (self, asset, bid, ask, size):
		'''
		Function: handle_quote_placement
		Purpose: Keeps track of active orders and updates them if its possible to
		Params: A Quote's bid, ask, size, and asset
		'''
		await self.place_bid_update(asset, size, bid, self.bid_order_ids[asset])
		await self.place_ask_update(asset, size, ask, self.bid_order_ids[asset])

	'''
	-------------------------------------------------- Options Pricing Functions --------------------------------------------------
	'''
	def option_specific_IV (self, strike, flag):
		'''
		Function: option_specific_IV
		Purpose: Calculates market implied vol for a specific asset
		Params: strike price and flag (c or p)
		'''
		price = self.current_mrkt_price[f"UC{strike}{flag}"]
		S = self.current_mrkt_price["UC"]
		K = strike
		t = (26 - self.current_day) / 252
		r = 0
		flag_low = flag.lower()
		try:
			sigma = py_vollib.black_scholes.implied_volatility.implied_volatility(price, S, K, t, r, flag_low)
		except:
			print('py vollib error')
			sigma = .2
		return sigma


	def garch_vol (self, strike, flag):
		'''
		Function: garch_vol
		Params: strike price and flag (c or p)
		'''
		if self.Garch_day > 23:
			am = arch_model(self.rolling_GARCH['returns'], vol="Garch", p=1, o=0, q=1, dist="normal", rescale=True) #rescale by 10y
			res = am.fit(update_freq=5)
			forecast = res.forecast(horizon=1, reindex=False).variance
			fcast = forecast.iloc[0]
			fcast = (fcast/10) ** 0.5 #scale back down
			print(fcast[0])
			return fcast[0]
		else:
			return 0.2

	def vol_skew(self):
        #this is the cvol calculation
		diff_c = 100
		diff_p = 100
		d_25_call = 0
		d_25_put = 0

		S = self.underlying_price

		t = (26 - self.current_day) / 252

		for strike in option_strikes:           #find strike with abs(option delta) closest to 25
			for flag in ["C", "P"]:
				sigma = self.option_specific_IV(strike, flag)
				if flag == "C":
					option_delta = py_vollib.black_scholes.greeks.analytical.delta(flag.lower(), S, strike,t,0,sigma)
					option_delta = option_delta * 100
					if abs(option_delta - 25) < diff_c:
						diff_c = abs(option_delta - 25)
						d_25_call = strike

				if flag == "P":
					option_delta = py_vollib.black_scholes.greeks.analytical.delta(flag.lower(), S, strike,t,0,sigma)
					option_delta = option_delta * 100
					if abs(25 + option_delta) < diff_p:
						diff_p = abs(option_delta - 25)
						d_25_put = strike
		cvol = self.option_specific_IV(d_25_call, "C") - self.option_specific_IV(d_25_put, "P")
		return cvol

	def black_scholes_price (self, S, K, t, r, flag):
		'''
		Function: black_scholes_price
		Purpose: Calculates the black scholes price based on the average of yang zhang vol and IV
		Params: spot, strike, time to expiry, risk free rate, flag (c or p)
		'''
		IV = self.option_specific_IV(K, flag)
		gvol = self.garch_vol(K, flag)
		cvol = self.vol_skew()
		if cvol > 0:
			if flag == "C":
				gvol += cvol/2
			if flag == "P":
				gvol -= cvol/2
		if cvol < 0:
			if flag == "C":
				gvol -= cvol/2
			if flag == "P":
				gvol += cvol/2
		if IV == None:
			return None
		sigma = (IV + gvol)/2 # Weight of IV vs YZ
		return py_vollib.black_scholes.black_scholes(flag.lower(),S,K,t,r,sigma)

	'''
	-------------------------------------------------- Rate/Greek Limit Functions --------------------------------------------------
	'''
	async def hedge_delta (self):
		vega_weighted_IV = self.vega_weighted_exchange_IV()
		total_delta = 0
		for strike in option_strikes:
			for flag in ["C", "P"]:
				position_size = self.positions[f"UC{strike}{flag}"]
				S = self.current_mrkt_price["UC"]
				K = strike
				t = (26 - self.current_day) / 252
				r = 0
				sigma = vega_weighted_IV
				option_delta = py_vollib.black_scholes.greeks.analytical.delta(flag.lower(), S, K,t,r,sigma)
				total_delta += (option_delta * position_size * 100)
		underlying_adjusted_delta = total_delta + self.positions['UC']
		print('Adjusted Delta: ', underlying_adjusted_delta, 'Raw Delta: ', total_delta, 'Underlying: ', self.positions['UC'])
		size = abs(int(underlying_adjusted_delta))
		if size == 0:
			return
		# print('PLACING NEW DELTA ORDER OF SIZE {} in direction ', size, (underlying_adjusted_delta > 0))
		if self.delta_order == None: # place new order
			if underlying_adjusted_delta > 0:
				ask_response = await self.place_order('UC', pb.OrderSpecType.MARKET, pb.OrderSpecSide.ASK, size)
				self.delta_order = ask_response.order_id
			elif underlying_adjusted_delta < 0:
				bid_response = await self.place_order('UC', pb.OrderSpecType.MARKET, pb.OrderSpecSide.BID, size)
				self.delta_order = bid_response.order_id
		else:
			if underlying_adjusted_delta > 0:
				ask_response = await self.modify_order(self.delta_order, 'UC', pb.OrderSpecType.MARKET, pb.OrderSpecSide.ASK, size)
				self.delta_order = ask_response.order_id
			elif underlying_adjusted_delta < 0:
				bid_response = await self.modify_order(self.delta_order, 'UC', pb.OrderSpecType.MARKET, pb.OrderSpecSide.BID, size)
				self.delta_order = bid_response.order_id


	def vega_weighted_exchange_IV (self):
		'''
		Function: vega_weighted_exchange_IV
		Purpose: Computes the IV of the chain at an instant
		Params: None
		'''
		total_vol = 0
		total_weight = 0

		for strike in option_strikes:
			for flag in ["C", "P"]:
				price = self.current_mrkt_price[f"UC{strike}{flag}"]
				S = self.current_mrkt_price["UC"]
				K = strike
				t = (26 - self.current_day) / 252
				r = 0
				flag_low = flag.lower()

				weight = math.exp(0.5 * math.log(price / strike) ** 2)

				try:
					vol = py_vollib.black_scholes.implied_volatility.implied_volatility(price, S, K, t, r, flag_low)
				except:
					print('py vollib error')
					vol = 0
					weight = 0
				total_vol += weight * vol
				total_weight += weight

		exchange_vol_estimate = total_vol / total_weight

		return exchange_vol_estimate

if __name__ == "__main__":
	start_bot(goofygoober)
