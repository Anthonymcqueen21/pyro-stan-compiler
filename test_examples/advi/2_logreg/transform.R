#!/usr/bin/env Rscript
#library(rjson)
source("model.data.R")
#x <- lapply(ls(), function(x) c(x,get(x)))
#j <- toJSON(x)
#write(j, "model.data.json")
Y <- ls()
library(RJSONIO)
list1 <- vector(mode="list", length=2)
list1[[1]] <- Y
list1[[2]] <- lapply(Y, function(x) get(x))

exportJson <- toJSON(list1)
write(exportJson, "model.data.json")

