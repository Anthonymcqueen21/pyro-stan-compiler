#!/usr/bin/env Rscript
#library(rjson)
args = commandArgs(trailingOnly=TRUE)
# test if there is at least one argument: if not, return an error
if (length(args)!=2) {
  stop("Two arguments needed (input R data file, output JSON file).n", call.=FALSE)
}
# e.g. args[1] = "model.data.R"
source(args[1])
Y <- ls()
Y <- Y[Y != "args"]
library(RJSONIO)
list1 <- vector(mode="list", length=2)
list1[[1]] <- Y
list1[[2]] <- lapply(Y, function(v__) get(v__))
exportJson <- toJSON(list1)
# e.g. args[2] = "model.data.json"
write(exportJson, args[2])

