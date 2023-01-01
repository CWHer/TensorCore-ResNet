/** @file common.cpp
*/
#include "common.h"

[[maybe_unused]] int randomInt(int min, int max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(min, max);
  return dist(gen);
}

[[maybe_unused]] float randomFloat(float min, float max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(min, max);
  return dist(gen);
}

void SimpleTimer::start(const std::string &name) {
  auto &timing_item = timing_items[name];
  timing_item.last_time = std::chrono::system_clock::now();
}

void SimpleTimer::end(const std::string &name) {
  auto &timing_item = timing_items[name];
  auto duration = std::chrono::system_clock::now() - timing_item.last_time;
  timing_item.total_duration += duration;
  timing_item.count++;
}

void SimpleTimer::printStat(const std::string &name) {
  auto &timing_item = timing_items[name];
  std::cout << name << " = "
            << timing_item.total_duration.count() / timing_item.count
            << " ms" << std::endl;
}
