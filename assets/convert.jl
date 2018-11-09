using BSON
using Flux, FluxJS

BSON.@load "m.bson" m
Flux.testmode!(m)
FluxJS.compile("model", m, rand(99))