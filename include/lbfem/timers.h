// Wall time per stage of a time step (for a traffic or flop model). The timers
// are local to each MPI rank: timing a stage adds no synchronization to the
// time loop. The totals are reduced over the ranks once, at the end
// (max_wall_times).
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

    // The accumulated time of each stage and, last, the rest of total (this
    // rank's time of the steps), each the maximum over the ranks of comm.
    std::array<double, n_stages + 1>
    max_wall_times(const double total, const MPI_Comm comm) const
    {
      std::array<double, n_stages + 1> t;
      t[n_stages] = total;
      for (unsigned int s = 0; s < n_stages; ++s)
        {
          t[s] = timers[s].wall_time();
          t[n_stages] -= t[s];
        }
      dealii::Utilities::MPI::max(dealii::ArrayView<const double>(t), comm, dealii::ArrayView<double>(t));
      return t;
    }

  private:
    std::array<dealii::Timer, n_stages> timers;
  };
} // namespace lbfem
