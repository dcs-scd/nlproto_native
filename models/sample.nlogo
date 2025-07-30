NetLogo 6.4.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
GRAPHICS-WINDOW
210
10
649
470
-1
-1
13.0
1
10
1
1
1
0
1
1
0.0
0.0
1.0
0.0
1
30
-16
16
-11
11
0
0
1
ticks

@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
0
@#$#@#$#@
@#$#@#$#@
0.0
0.0
0.0
0.0
0
0
0
0
NIL

@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
0
@#$#@#$#@
0.0
0.0
0.0
0.0
0
0
0
T
NIL

@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
0
@#$#@#$#@
0.0
0.0
0.0
0.0
0
0
0
T
NIL

@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
0
@#$#@#$#@
0.0
0.0
0.0
0.0
0
0
0
T
NIL

@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
0
@#$#@#$#@
0.0
0.0
0.0
0.0
0
0
0
T
NIL

@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
0
@#$#@#$#@
0.0
0.0
0.0
0.0
0
0
0
T
NIL

@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
0
false
Polygon -7500403 true true 150 5 40 250 230 250

@#$#@#$#@
@#$#@#$#@
to setup
  clear-all
  reset-ticks
  
  create-turtles 50 [
    setxy random-xcor random-ycor
    set color (random 140) + 5
    set size 2
  ]
end

to go
  if ticks >= 500 [ stop ]
  
  ask turtles [
    rt random 60 - 30
    fd 0.5
    if abs xcor > 15 or abs ycor > 15 [
      setxy random-xcor random-ycor
    ]
  ]
  
  let mean-color mean [color] of turtles
  set-current-plot "Tick-Statistics"
  set-current-plot-pen "pop"
  plot count turtles
  set-current-plot-pen "mean"
  plot mean-color
  
  tick
end

@#$#@#$#@
@#$#@#$#@
@#$#@#$#@

1

@#$#@#$#@
@#$#@#$#@
@#$#@#$#@

setup
go

@#$#@#$#@
@#$#@#$#@
Plot
Tick-Statistics
-500
500
0.0
200.0
true
false
"" ""
PENS
"pop" 1.0 0 -16777216 true "" ""
"mean" 1.0 0 -13345367 true "" ""

@#$#@#$#@
@#$#@#$#@
NIL
1