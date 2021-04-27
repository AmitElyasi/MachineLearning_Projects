#################################
# Your name: Amit Elyasi 316291434
#################################

import math
import numpy as np
import matplotlib.pyplot as plt


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """
    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """

        xs = np.random.uniform(0, 1, size=m)
        xs.sort()
        ys = np.zeros(m)
        for i in range(m):
            if xs[i] <= 0.2 or (0.4 <= xs[i] <= 0.6) or xs[i] >= 0.8:
                ys[i]= np.random.choice([0, 1], p=[0.2, 0.8])
            else:
                ys[i] = np.random.choice([0, 1], p=[0.9, 0.1])
        return np.array([xs, ys])

    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """
        sample = self.sample_from_D(m)
        xs = sample[0]
        ys = sample[1]
        plt.xlabel("x")
        plt.ylabel("labels")
        plt.plot(xs, ys, 'o', color='blue')
        plt.ylim((-0.1, 1.1))
        line = [0.2, 0.4, 0.6, 0.8]
        for x in line:
            plt.axvline()
            plt.axvline(x=x, color="black", linestyle=':')
        #plt.show()

        interval_set, error = find_best_interval(xs, ys, k)
        for interval in interval_set:
            points = np.linspace(interval[0], interval[1], num=1000)
            plt.plot(points, [-0.1] * 1000, linewidth=6)
        #plt.show

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        m_range = range(m_first, m_last + 1, step)

        true_error_avg = []
        empirical_error_avg = []

        for m in m_range:
            true_error = 0
            empirical_error = 0
            for t in range(T):
                sample = self.sample_from_D(m)
                interval_set, error = find_best_interval(sample[0],sample[1], k)
                empirical_error += self.calculate_empirical_error(interval_set,sample[0],sample[1])
                true_error += self.calculate_true_error(interval_set)

            true_error_avg.append(true_error/T)
            empirical_error_avg.append(empirical_error/T)

        plt.plot(m_range, true_error_avg, marker='o', color='red', label='True Error')
        plt.plot(m_range, empirical_error_avg, marker='o', color='blue', label='Empirical Error')
        plt.legend()
        #plt.show()

        return np.array([empirical_error_avg, true_error_avg])

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        sample = self.sample_from_D(m)
        X, Y = sample[0], sample[1]
        empirical_error = []
        true_error = []
        k_range = range(k_first, k_last+1, step)

        for k in k_range:
            interval_set, error = find_best_interval(X, Y, k)
            empirical_error.append(self.calculate_empirical_error(interval_set, X, Y))
            true_error.append(self.calculate_true_error(interval_set))

        plt.plot(k_range, true_error, marker='o', color='red', label='True Error')
        plt.plot(k_range, empirical_error, marker='o', color='blue', label='Empirical Error')
        plt.legend()
        #plt.show()

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        sample = self.sample_from_D(m)
        X, Y = sample[0], sample[1]
        empirical_error = []
        true_error = []
        penalty = []
        sum_penalty_emp = []
        k_range = range(k_first, k_last+1, step)

        for k in k_range:
            interval_set, error = find_best_interval(X, Y, k)
            empirical_error.append(self.calculate_empirical_error(interval_set, X, Y))
            true_error.append(self.calculate_true_error(interval_set))
            penalty.append(self.penalty(k, m, 0.1))
            sum_penalty_emp.append(penalty[-1]+empirical_error[-1])

        plt.plot(k_range, true_error, marker='o', color='red', label='True Error')
        plt.plot(k_range, empirical_error, marker='o', color='blue', label='Empirical Error')
        plt.plot(k_range, penalty, marker='o', color='green', label='Penalty')
        plt.plot(k_range, sum_penalty_emp, marker='o', color='yellow', label='Sum Of Penalty and Empirical Error')
        plt.legend()
        #plt.show()


    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """

        k_range = range(1, 11)
        holdout_error = np.array([0.0 for k in range(1, 31)])
        holdout_error = np.reshape(holdout_error, (3, 10))
        minimal_k = [0, 0, 0]
        for t in range(T):
            sample = self.sample_from_D(m)
            np.random.shuffle(sample)
            train = [(sample[0][i], sample[1][i]) for i in range(1200)]
            train.sort(key=lambda x: x[0])
            holdout = [(sample[0][i], sample[1][i]) for i in range(1201, 1500)]
            train_xs = np.array([x[0] for x in train])
            train_ys = np.array([x[1] for x in train])
            test_xs = np.array([x[0] for x in holdout])
            test_ys = np.array([x[1] for x in holdout])
            for k in k_range:
                intervals_set, error = find_best_interval(train_xs, train_ys, k)
                holdout_error[t][k-1] = self.calculate_empirical_error( intervals_set, test_xs, test_ys)
            minimal_k[t] = np.argmin(holdout_error[t]) + 1
        k = min(minimal_k)
        plt.plot(k_range, holdout_error[0], 'r', k_range, holdout_error[1], 'b', k_range, holdout_error[2], 'g')
        #plt.show()
        return k

    #################################
    # Place for additional methods

    def find_best_interval(xs, ys, k):
    assert all(array(xs) == array(sorted(xs))), "xs must be sorted!"

    xs = array(xs)
    ys = array(ys)
    m = len(xs)
    P = [[None for j in range(k+1)] for i in range(m+1)]
    E = zeros((m+1, k+1), dtype=int)
    
    # Calculate the cumulative sum of ys, to be used later
    cy = concatenate([[0], cumsum(ys)])
    
    # Initialize boundaries:
    # The error of no intervals, for the first i points
    E[:m+1,0] = cy
    
    # The minimal error of j intervals on 0 points - always 0. No update needed.        
        
    # Fill middle
    for i in range(1, m+1):
        for j in range(1, k+1):
            # The minimal error of j intervals on the first i points:
            
            # Exhaust all the options for the last interval. Each interval boundary is marked as either
            # 0 (Before first point), 1 (after first point, before second), ..., m (after last point)
            options = []
            for l in range(0,i+1):  
                next_errors = E[l,j-1] + (cy[i]-cy[l]) + concatenate([[0], cumsum((-1)**(ys[arange(l, i)] == 1))])
                min_error = argmin(next_errors)
                options.append((next_errors[min_error], (l, arange(l,i+1)[min_error])))

            E[i,j], P[i][j] = min(options)
    
    # Extract best interval set and its error count
    best = []
    cur = P[m][k]
    for i in range(k,0,-1):
        best.append(cur)
        cur = P[cur[0]][i-1]       
        if cur == None:
            break 
    best = sorted(best)
    besterror = E[m,k]
    
    # Convert interval boundaries to numbers in [0,1]
    exs = concatenate([[0], xs, [1]])
    representatives = (exs[1:]+exs[:-1]) / 2.0
    intervals = [(representatives[l], representatives[u]) for l,u in best]

    return intervals, besterror


    def intersection(self, intervals_set, I):
        """
        returns the intersection of the intervals with the interval I
        """
        sum = 0
        for interval in intervals_set:
            a = max(I[0], interval[0])
            b = min(I[1], interval[1])
            if a < b:
                sum += (b - a)
        return sum

    def union(self, interval_set):
        union = 0
        for interval in interval_set:
            union += interval[1] - interval[0]
        return union

    def find_complements(self, intervals):
        insert(0, (0, 0))
        append((1, 1))
        complements = []
        for i in range(len(intervals) - 1):
            complements.append((intervals[i][1], intervals[i + 1][0]))
        return complements
    

    def calculate_true_error(self, intervals_set):
        """
        X1 = [0,0.2]U[0.4,0.6]U[0.8,1]
        X2 = [0.2,0.4]U[0.6,0.8]
        returns the true error of h_(interval_set)
        """

        sumX1 = 0
        sumX2 = 0
        complements_set = self.find_complements(intervals_set)
        union_complements = self.union(complements_set)
        union_intervals = self.union(intervals_set)

        sumX1 += self.intersection(intervals_set, [0, 0.2])
        sumX1 += self.intersection(intervals_set, [0.4, 0.6])
        sumX1 += self.intersection(intervals_set, [0.8, 1])
        expectation = sumX1 * 0.2 + (union_intervals-sumX1) * 0.9

        sumX2 += self.intersection(complements_set, [0.2, 0.4])
        sumX2 += self.intersection(complements_set, [0.6, 0.8])
        expectation += sumX2 * 0.1 + (union_complements-sumX2) * 0.8

        return expectation

    def calculate_empirical_error(self, intervals_set, X, Y):
        """
        returns the empirical error of h_interval_set
        """
        error_count = 0
        for i in range(len(X)):
            y = 0
            for interval in intervals_set:

                a = interval[0]
                b = interval[1]
                if a <= X[i] <= b:
                    y = 1
                    break
                if X[i] < a:  # intervals are sorted
                    break
            if y != Y[i]:
                error_count += 1

        return error_count / len(X)

    def penalty(self, k, m, delta):
        a = 8 / m
        in_log = (m * math.exp(1)) / k
        b = 2 * k * math.log(in_log)
        c = math.log(4 / delta)
        res = a * (b + c)

        return math.sqrt(res)

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.draw_sample_intervals(100, 3)
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500, 3)

