# Vector plot of the Taylor-Green vortex computed by cgdbe.
#
#   gnuplot plot_vortex.gp                                  -> vortex.png
#   gnuplot -e "file='vortex_initial.dat'; out='t0.png'" plot_vortex.gp
#
# Data: "x y u/U0 v/U0 |u|/U0", one blank-line separated block per grid row.

if (!exists("file"))   file   = "vortex_final.dat"
if (!exists("out"))    out    = "vortex.png"
if (!exists("arrows")) arrows = 16        # approx. number of arrows per direction

set terminal pngcairo size 1500,700 font "Helvetica,12"
set output out

# thin out the nodes and scale the longest arrow to ~ one arrow spacing
stats file using 5 nooutput
rows   = STATS_blank + 1
umax   = STATS_max
stride = (rows - 1) / arrows > 1 ? int((rows - 1) / arrows) : 1
len    = 0.9 * stride / (rows - 1.) / umax

header = system(sprintf("head -n 1 %s | cut -c 3-", file))

set multiplot layout 1,2

# --- velocity vectors, coloured by |u|/U0 -----------------------------------
set title "Taylor-Green vortex, CGDBE:  ".header noenhanced
set size ratio -1
set xrange [-0.03:1.03]; set yrange [-0.03:1.03]
set xlabel "x/L"; set ylabel "y/L"
set cblabel "|u| / U_0"
set cbrange [0:*]
set palette defined (0 "#9ecae1", 0.35 "#3182bd", 0.7 "#54278f", 1 "#cb181d")
unset key
plot file every stride:stride \
     using ($1 - 0.5*len*$3):($2 - 0.5*len*$4):(len*$3):(len*$4):5 \
     with vectors head filled size screen 0.008,20 lw 1.6 lc palette

# --- kinetic energy decay and velocity error ---------------------------------
set title "Kinetic energy decay and velocity error"
set size noratio
set xrange [*:*]; set yrange [*:*]; set y2range [*:*]
set xlabel "t  (lattice units)"
set ylabel "E(t) / E(0)"
set y2label "relative L_2 error of u"
set ytics nomirror; set y2tics; set logscale y2; set format y2 "10^{%T}"
set key bottom left box opaque
plot "vortex_history.dat" using 1:3 with lines lw 2 lc rgb "black" title "exact, exp(-4{/Symbol n}k^2t)", \
     ""                using 1:2 with points pt 6 ps 1.3 lc rgb "#d62728" title "CGDBE", \
     ""                every ::1 using 1:4 axes x1y2 with linespoints pt 5 ps 0.7 lc rgb "#1f77b4" title "error (right axis)"

unset multiplot
