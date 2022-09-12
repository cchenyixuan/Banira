import numpy as np
from multiprocessing import Pool


class Strategy:
    def __init__(self):
        self.profit = 0.0
        self.trade_times = 0
        self.plus_times = 0
        self.last_price = 0.0
        self.status = 0
        self.long_history = []
        self.short_history = []
        self.empty_history = []

    def _long(self, price, i):
        self.status = 1
        self.long_history.append(i)
        self.last_price = price

    def _short(self, price, i):
        self.status = -1
        self.short_history.append(i)
        self.last_price = price

    def _empty(self, price, i):
        delta = (price - self.last_price) / self.last_price
        self.profit += self.status*delta  # plus if long else minus
        self.profit -= 0.0008  # commission fee
        self.status = 0
        self.empty_history.append(i)

    def long(self, price, i):
        if self.status == 0:
            self._long(price, i)
        elif self.status == -1:
            self._empty(price, i)
            self._long(price, i)

    def short(self, price, i):
        if self.status == 0:
            self._short(price, i)
        elif self.status == 1:
            self._empty(price, i)
            self._short(price, i)

    def empty(self, price, i):
        if self.status != 0:
            self._empty(price, i)


class LinearRegression:
    def __init__(self, data, length, interval):
        self.data = data
        self.length = length
        self.interval = interval
        self.strategy = Strategy()

    def regression_cal(self, length):
        derivative = np.zeros_like(self.data)
        for i in range(0, self.data.shape[0], self.interval):
            if i < length:
                continue
            x = np.array(range(length))
            y = self.data[i - length + 1:i + 1]
            b1 = (np.sum(y * x) - length * np.mean(x) * np.mean(y)) / (
                        np.sum(x * x) - length * np.mean(x) * np.mean(x))
            # b0 = np.mean(y) - b1 * np.mean(x)
            derivative[i] = b1
        return derivative

    def worker(self, length):
        print(length)
        return self.regression_cal(length)

    def multip(self):
        pool = Pool()
        arguments = [i for i in range(5, 121, 1)]
        self.ans = pool.map(self.worker, arguments)
        lr = np.array(self.ans).T
        rate = np.zeros((lr.shape[0], 7), dtype=np.float32)

        for i in range(lr.shape[0]):
            long_term_plus = len([item for item in lr[i, lr.shape[1] // 2:] if item > 0])
            short_term_plus = len([item for item in lr[i, :lr.shape[1] // 2] if item > 0])
            long_term_rate = long_term_plus/(lr.shape[1] // 2)
            short_term_rate = short_term_plus / (lr.shape[1] // 2)
            average_rate = (long_term_plus+short_term_plus)/lr.shape[1]
            plus_depth = ([item if item > 0 else None for item in lr[i, :]]+[None]).index(None)
            minus_depth = ([item if item < 0 else None for item in lr[i, :]]+[None]).index(None)
            norm = sum([abs(_) for _ in lr[i, :]])
            std = np.std(lr[i, :])
            rate[i] = np.array((short_term_rate, long_term_rate, average_rate, norm, std, plus_depth, minus_depth), dtype=np.float32)
        np.save("_lr.npy", lr)
        np.save("_rate.npy", rate)
        np.save("_data.npy", np.hstack((np.reshape(self.data, (self.data.shape[0], 1)), lr)))

    def regression(self):
        series = []
        series_x = []
        start = self.data[0]
        for i in range(0, self.data.shape[0], self.interval):
            if i < self.length:
                continue
            x = np.array(range(self.length))
            y = self.data[i - self.length + 1:i + 1]
            b1 = (np.sum(y * x) - self.length * np.mean(x) * np.mean(y)) / (
                        np.sum(x * x) - self.length * np.mean(x) * np.mean(x))
            b0 = np.mean(y) - b1 * np.mean(x)
            series.extend([b0 + z * b1 for z in [0, self.length - 1]])
            series_x.extend([i - self.length + 1, i])
        print(len(series))
        return series_x, series

    def regression_strategy(self):
        heads = np.zeros((self.data.shape[0], 2), dtype=np.float32)
        tails = np.zeros((self.data.shape[0], 2), dtype=np.float32)
        derivative = lambda index: (heads[index][1] - tails[index][1]) / (heads[index][0] - tails[index][0])
        for i in range(0, self.data.shape[0], self.interval):
            if i < self.length:
                continue
            x = np.array(range(self.length))
            y = self.data[i - self.length + 1:i + 1]
            b1 = (np.sum(y * x) - self.length * np.mean(x) * np.mean(y)) / (
                        np.sum(x * x) - self.length * np.mean(x) * np.mean(x))
            b0 = np.mean(y) - b1 * np.mean(x)
            tails[i] = np.array([i - self.length + 1, b0], dtype=np.float32)
            heads[i] = np.array([i, b0 + (self.length - 1) * b1], dtype=np.float32)
            if i < self.length * 2:
                continue
            # strategy part
            if derivative(i) > derivative(i - 1) and abs(derivative(i)) < 4:
                self.strategy.long(self.data[i], i)
            elif derivative(i) < derivative(i - 1) and abs(derivative(i)) < 4:
                self.strategy.short(self.data[i], i)

        print(self.strategy.profit)


class LinearRegressionStrategy(Strategy):
    def __init__(self):
        super().__init__()
        self.buffer = np.load("GPUWorkflow/_data.npy")
        self.close = self.buffer[:, 0]
        self.lr = self.buffer[:, 1:]
        self.rate = np.load("GPUWorkflow/_rate.npy")

    def __call__(self, long_threshold=0.5, short_threshold=0.5, average_threshold=0.6, norm_threshold=1000, std_threshold=100, depth_threshold=5):
        for i in range(self.buffer.shape[0]-200):
            if i < 200:#self.lr.shape[1]:
                continue
            if self.rate[i][0] > short_threshold and self.rate[i][1] > long_threshold and self.rate[i][2] > average_threshold and self.rate[i][3]>norm_threshold and self.rate[i][4]>std_threshold and self.rate[i][5]>depth_threshold:
                self.long(self.close[i], i)
            elif self.rate[i][0] < 1-short_threshold and self.rate[i][1] < 1-long_threshold and self.rate[i][2] < 1-average_threshold and self.rate[i][3]>norm_threshold and self.rate[i][4]>std_threshold and self.rate[i][6]>depth_threshold:
                self.short(self.close[i], i)

            elif self.rate[i][3]<norm_threshold and self.rate[i][4]<std_threshold:
                self.empty(self.close[i], i)
        print(self.profit)
        # return self.profit, long_threshold, short_threshold, average_threshold, norm_threshold, std_threshold
        #
        import matplotlib.pyplot as plt
        plt.plot(self.close)
        plt.plot(self.long_history,  [self.close[i] for i in self.long_history], "r^")
        plt.plot(self.short_history, [self.close[i] for i in self.short_history], "gv")
        plt.plot(self.empty_history, [self.close[i] for i in self.empty_history], "y*")
        plt.show()


if __name__ == "__main__":
    from binance.client import Client

    # client = Client("dBw27g6LP3rU3bAbGm7iHrxtsvzTR9KSL3GVcRa8ZQdqMhgCh4uWU149IEo52X5c",
    #                 "sQJfbMJLiYqQ5eQPXpgNiKuH7yAcVDf8xv5hTbvsv5ehVGWHbigIrxrOtDQcMnfR")
    # close_buffer = np.array(
    #     client.futures_historical_klines(symbol="BTCUSDT", interval="1m", start_str="30 day ago UTC"),
    #     dtype=np.float32
    # )[:, 4]
    # data = close_buffer
    data = np.load("_90data.npy")[:, 0]
    lr = LinearRegression(data, 5, 1)
    lr.multip()
    a = LinearRegressionStrategy()
    a(0.99, 0.0, 0.98, 0, 10, 10)
    # for i in range(100):
    #     print(i)
    #     a = LinearRegressionStrategy()
    #     a(0.5 + 0.5 / 100 * i, 0.5 - 0.5 / 100 * i)
