// Wall time per stage of a time step, both as TimerOutput sections (for the
// summary table) and as accumulated Timers (for a traffic or flop model).
#pragma once

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
    static constexpr std::array<const char *, n_stages> names = {
      {"1 collision (nodal)", "2 advection (cell/face loop)", "3 mass solves"}};

    explicit StageTimers(dealii::TimerOutput &output)
      : output(output)
    {
      for (auto &t : timers)
        t.stop();
    }

    // Times its lifetime in both the TimerOutput section and the stage Timer.
    class Scope
    {
    public:
      Scope(StageTimers &st, const Stage stage)
        : section(st.output, names[stage])
        , timer(st.timers[stage])
      {
        timer.start();
      }
      ~Scope()
      {
        timer.stop();
      }

    private:
      dealii::TimerOutput::Scope section;
      dealii::Timer             &timer;
    };

    // Runs fn() within the given stage and returns its result.
    template <std::invocable F>
    decltype(auto)
    time(const Stage stage, F &&fn)
    {
      const Scope scope(*this, stage);
      return std::invoke(std::forward<F>(fn));
    }

    double
    wall_time(const Stage stage) const
    {
      return timers[stage].wall_time();
    }

  private:
    dealii::TimerOutput                   &output;
    std::array<dealii::Timer, n_stages> timers;
  };
} // namespace lbfem
