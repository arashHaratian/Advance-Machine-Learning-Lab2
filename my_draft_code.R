library(HMM)
library(entropy)


# 1 -----------------------------------------------------------------------


n_states <- 10
states <- 1:n_states
symbols <- letters[states]
start_probs <- rep(1/n_states, n_states)

transition_probs <- diag(0.5, 10)
for(i in seq_len(n_states)){
  j <- ifelse(i+1 <= n_states, i+1, 1)
  transition_probs[i, j] <- 0.5
}


emission_probs <- matrix(0, nrow = 10, ncol = 10)
for(i in seq_len(n_states)){
  j <- (i-2):(i+2)
  j[j <= 0] <- 10 + j[j <= 0]
  j[j > 10] <- j[j > 10] - 10
  emission_probs[i, j] <- 0.2
}




robot_HMM <- initHMM(states, symbols, start_probs, transition_probs, emission_probs)


# 2 -----------------------------------------------------------------------
set.seed(1378)
robot_sim <- simHMM(robot_HMM, 100)


# 3 -----------------------------------------------------------------------
robot_observations <- robot_sim$observation

alpha <- exp(forward(robot_HMM, robot_observations))
beta <- exp(backward(robot_HMM, robot_observations))

# smoothing_dist <- (alpha * beta) / colSums(alpha * beta)
smoothing_dist <- prop.table(alpha * beta, 2)
# filtering_dist <- alpha / colSums(alpha)
filtering_dist <- prop.table(alpha, 2)

likely_path <- viterbi(robot_HMM, robot_observations)

# 4 -----------------------------------------------------------------------


smoothing_path <- apply(smoothing_dist, 2, which.max)
filtering_path <- apply(filtering_dist, 2, which.max)

table(smoothing_path, robot_sim$states)
mean(smoothing_path == robot_sim$states)

table(filtering_path, robot_sim$states)
mean(filtering_path == robot_sim$states)

table(likely_path, robot_sim$states)
mean(likely_path == robot_sim$states)

# 5 -----------------------------------------------------------------------
set.seed(1343)
robot_sim <- simHMM(robot_HMM, 100)
robot_observations <- robot_sim$observation

alpha <- exp(forward(robot_HMM, robot_observations))
beta <- exp(backward(robot_HMM, robot_observations))

# smoothing_dist <- (alpha * beta) / colSums(alpha * beta)
smoothing_dist <- prop.table(alpha * beta, 2)
# filtering_dist <- alpha / colSums(alpha)
filtering_dist <- prop.table(alpha, 2)
likely_path <- viterbi(robot_HMM, robot_observations)

smoothing_path <- apply(smoothing_dist, 2, which.max)
filtering_path <- apply(filtering_dist, 2, which.max)

table(smoothing_path, robot_sim$states)
mean(smoothing_path == robot_sim$states)

table(filtering_path, robot_sim$states)
mean(filtering_path == robot_sim$states)

table(likely_path, robot_sim$states)
mean(likely_path == robot_sim$states)

# More accurate than filtering as we have the whole data points and we can use both the past and the future to do inference on value of zt
# The case is also is true for Viterbi result. The algorithm maximizes the value of zt given all the past values of z and x.


# 6 -----------------------------------------------------------------------
filter_entropy <- apply(filtering_dist, 2, entropy.empirical)
plot(filter_entropy, type = "l")

smoothing_entropy <- apply(smoothing_dist, 2, entropy.empirical)
plot(smoothing_entropy, type = "l")



# 7 -----------------------------------------------------------------------

# p(zt+1 | z0:T, x0:T) = p(zt+1|zT)p(zT|x0:T)
# p(zt+1|zt)  is transition_probs
# p(zt+1|zt)  last filter 

filtering_dist[, 100] %*% transition_probs
