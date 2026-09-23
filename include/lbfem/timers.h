// Wall time per stage of a time step (for a traffic or flop model). The timers
// are local to each MPI rank: timing a stage adds no synchronization to the
// time loop. Reduce the totals over ranks once, at the end (max_wall_times).
#pragma once

#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

#include <array>
#include <concepts>
#include <functional>
#include <utility>

namespace lbfem
{
  class StageTimers
  {
  public:
    enum Stage : unsigned int
    {
      collision, // nodal work
      advection, // cell and face loops
      mass,      // mass-matrix solves
      n_stages
    };

    StageTimers()
    {
      for (auto &t : timers)
        t.stop();
    }

    // Times its lifetime in the stage timer.
    class Scope
    {
    public:
      Scope(StageTimers &st, const Stage stage)
        : timer(st.timers[stage])
      {
        timer.start();
      }
      ~Scope()
      {
        timer.stop();
      }

    private:
      dealii::Timer &timer;
    };

    // Runs fn() within the given stage and returns its result.
    template <std::invocable F>
    decltype(auto)
    time(const Stage stage, F &&fn)
    {
      const Scope scope(*this, stage);
      return std::invoke(std::forward<F>(fn));
    }

    // The accumulated time of each stage on this rank.
    double
    wall_time(const Stage stage) const
    {
      return timers[stage].wall_time();
    }

  private:
    std::array<dealii::Timer, n_stages> timers;
  };
} // namespace lbfem
