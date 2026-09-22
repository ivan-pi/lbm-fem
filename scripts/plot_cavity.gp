# Streamlines of the lid-driven cavity: contours of the stream function written
# by  cgdbe --case cavity.
#
#   gnuplot plot_cavity.gp                                    # all cavity_Re*.dat in .
#   gnuplot -e "dir='data/cavity'" scripts/plot_cavity.gp     # the reference results
#   gnuplot -e "levels='ghia'" plot_cavity.gp                 # Ghia et al. (1982) levels
#   gnuplot -e "files='cavity_Re400.dat'; out='re400.png'" plot_cavity.gp
#
# Data: "x y u/U0 v/U0 psi/(U0 L) rho-rho0", one blank-line separated block per
# grid row (the grid may be non-uniform). Sign convention u = -dpsi/dy, i.e. the
# primary vortex is positive (Lee & Lin, Table I); Ghia et al. use the opposite
# sign, their levels are negated below.
#
# levels = 'leelin' (default): the levels of Fig. 5 of Lee & Lin (2001), which
#   depend on Re, plus a fixed set of secondary-vortex levels in every panel.
# levels = 'ghia': the 24 levels of Ghia, Ghia & Shin (1982), for all Re.

if (!exists("dir"))    dir    = "."
if (!exists("files"))  files  = system(sprintf("cd '%s' && ls cavity_Re*.dat 2>/dev/null | grep -v history | sort -t e -k 2 -n | sed 's|^|%s/|' | tr '\n' ' '", dir, dir))
if (!exists("out"))    out    = "cavity.png"
if (!exists("levels")) levels = "leelin"

n = words(files)
if (n == 0) { print "no cavity_Re*.dat files found"; exit }
cols = n < 3 ? n : 2
rows = (n + cols - 1) / cols

set terminal pngcairo size 620*cols,640*rows font "Helvetica,12"
set output out

# ---- contour levels ---------------------------------------------------------
# Lee & Lin, Fig. 5 (primary vortex); the secondary vortices are unlabelled in
# the paper, so a fixed set of small negative levels is added for them.
leelin_primary(Re) = Re <= 400  ? "0.11, 0.1, 0.09, 0.07, 0.05, 0.03, 0.01, 0.001" : \
                     Re <= 1000 ? "0.117, 0.11, 0.09, 0.06, 0.03, 0.01, 0.001" : \
                                  "0.12, 0.11, 0.09, 0.06, 0.03, 0.01, 0.001"
leelin_secondary = "-1e-6, -1e-5, -1e-4, -5e-4, -1e-3, -2e-3, -3e-3"

# Ghia, Ghia & Shin (1982), sign flipped to this code's convention.
ghia_primary   = "0.1175, 0.115, 0.11, 0.1, 0.09, 0.07, 0.05, 0.03, 0.01, 1e-4, 1e-5, 1e-7, 1e-10"
ghia_secondary = "-1e-8, -1e-7, -1e-6, -1e-5, -5e-5, -1e-4, -2.5e-4, -5e-4, -1e-3, -1.5e-3, -3e-3"

set contour base
unset surface
set cntrparam bspline
set cntrlabel onecolor

set size ratio -1
set xrange [0:1]; set yrange [0:1]
set xtics 0.25; set ytics 0.25
set xlabel "x/L"; set ylabel "y/L"
unset key

set multiplot layout rows,cols
do for [i = 1:n] {
  file = word(files, i)
  Re   = real(system(sprintf("head -n 1 %s | sed 's/.*Re = \\([0-9.]*\\).*/\\1/'", file)))
  psi  = system(sprintf("grep 'primary' %s | sed 's/.*psi = \\([-0-9.e]*\\).*/\\1/'", file))

  if (levels eq "ghia") {
    eval sprintf("set cntrparam levels discrete %s, %s", ghia_primary, ghia_secondary)
  } else {
    eval sprintf("set cntrparam levels discrete %s, %s", leelin_primary(Re), leelin_secondary)
  }

  set table $contours
  splot file using 1:2:5
  unset table

  set title sprintf("Re = %g,   psi_max = %s   (%s levels)", Re, psi, levels) noenhanced
  # primary vortex black, counter-rotating secondary vortices red
  plot $contours using 1:2:($3 > 0 ? 0x000000 : 0xcc2222) with lines lw 1.2 lc rgb variable
}
unset multiplot
