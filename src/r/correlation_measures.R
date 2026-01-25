
require(ggplot2)
require(ggthemes)

# ACE 
# https://en.wikipedia.org/wiki/Alternating_conditional_expectations
# https://github.com/partofthething/ace

library(acepack)
TWOPI <- 8 * atan(1)
x <- runif(200, 0, TWOPI)
y <- exp(sin(x) + rnorm(200)/2)
a <- ace(x, y)
par(mfrow=c(3,1))
plot(a$y, a$ty)  # view the response transformation
plot(a$x, a$tx)  # view the carrier transformation
plot(a$tx, a$ty) # examine the linearity of the fitted model


# https://search.r-project.org/CRAN/refmans/metrica/html/dcorr.html
# https://pypi.org/project/dcor/

require(metrica)
par(mfrow=c(1,1))
set.seed(1)
P <- rnorm(n = 100, mean = 0, sd = 10)
O <- P*P + rnorm(n=100, mean = 0, sd = 100)
dcorr(obs = P, pred = O)
plot(P,O)

