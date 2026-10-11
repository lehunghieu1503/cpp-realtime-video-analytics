#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>

// Samples the power drawn by the machine once a second on a background
// thread and averages it. What is measured depends on the platform:
//   - Jetson: whole-board input power from the INA3221 monitor (VDD_IN)
//   - x86: CPU package from Intel RAPL (covers the iGPU and NPU), plus the
//     NVIDIA GPU as reported by nvidia-smi when the backend runs on it
// watts() is negative when the machine exposes none of these. RAPL is
// readable only by root on most kernels; to allow it:
//   sudo chmod a+r /sys/class/powercap/intel-rapl:0/energy_uj
class PowerMeter {
public:
  // useNvidiaGpu: the inference backend runs on an NVIDIA GPU (cuda,
  // tensorrt). Otherwise the idle GPU would be counted against it.
  explicit PowerMeter(bool useNvidiaGpu) {
    namespace fs = std::filesystem;
    std::error_code ec;
    for (const auto& driver :
         fs::directory_iterator("/sys/bus/i2c/drivers/ina3221", ec)) {
      for (const auto& hwmon :
           fs::directory_iterator(driver.path() / "hwmon", ec)) {
        if (fs::exists(hwmon.path() / "in1_input", ec)) {
          jetsonDir_ = hwmon.path().string();
        }
      }
    }
    const char* rapl = "/sys/class/powercap/intel-rapl:0/energy_uj";
    if (jetsonDir_.empty() && std::ifstream(rapl).good()) raplPath_ = rapl;
    hasNvidiaSmi_ =
        useNvidiaGpu && jetsonDir_.empty() && readNvidiaSmi() >= 0.0;

    if (jetsonDir_.empty() && raplPath_.empty() && !hasNvidiaSmi_) return;
    running_ = true;
    thread_ = std::thread([this] { sampleLoop(); });
  }

  ~PowerMeter() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
  }

  // What the number covers, for the report
  std::string source() const {
    if (!jetsonDir_.empty()) return "board input (INA3221)";
    std::string s;
    if (!raplPath_.empty()) s = "CPU package (RAPL)";
    if (hasNvidiaSmi_) s += s.empty() ? "GPU (nvidia-smi)" : " + GPU (nvidia-smi)";
    return s.empty() ? "unavailable" : s;
  }

  // Forget the samples so far, e.g. after warm-up
  void reset() {
    sumWatts_ = 0.0;
    samples_ = 0;
  }

  // Mean since construction or the last reset(); negative if no sample yet
  double watts() const {
    int n = samples_;
    return n > 0 ? sumWatts_ / n : -1.0;
  }

  // Most recent sample; negative if none
  double lastWatts() const { return lastWatts_; }

private:
  static double readNumber(const std::string& path) {
    std::ifstream file(path);
    double value = -1.0;
    file >> value;
    return value;
  }

  static double readNvidiaSmi() {
    FILE* pipe = popen(
        "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits "
        "2>/dev/null",
        "r");
    if (!pipe) return -1.0;
    double watts = -1.0;
    if (fscanf(pipe, "%lf", &watts) != 1) watts = -1.0;
    pclose(pipe);
    return watts;
  }

  void sampleLoop() {
    using Clock = std::chrono::steady_clock;
    double lastEnergyUj = raplPath_.empty() ? 0.0 : readNumber(raplPath_);
    auto lastTime = Clock::now();

    while (running_) {
      for (int i = 0; i < 10 && running_; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      auto now = Clock::now();
      double seconds = std::chrono::duration<double>(now - lastTime).count();
      lastTime = now;

      double watts = 0.0;
      if (!jetsonDir_.empty()) {
        // millivolts x milliamps
        watts = readNumber(jetsonDir_ + "/in1_input") *
                readNumber(jetsonDir_ + "/curr1_input") / 1e6;
      }
      if (!raplPath_.empty()) {
        double energyUj = readNumber(raplPath_);
        // The counter wraps; skip that one sample
        if (energyUj >= lastEnergyUj) {
          watts += (energyUj - lastEnergyUj) / 1e6 / seconds;
        }
        lastEnergyUj = energyUj;
      }
      if (hasNvidiaSmi_) {
        double gpu = readNvidiaSmi();
        if (gpu > 0.0) watts += gpu;
      }
      lastWatts_ = watts;
      sumWatts_ = sumWatts_ + watts;
      samples_++;
    }
  }

  std::string jetsonDir_;
  std::string raplPath_;
  bool hasNvidiaSmi_ = false;
  std::atomic<bool> running_{false};
  std::atomic<double> sumWatts_{0.0};
  std::atomic<int> samples_{0};
  std::atomic<double> lastWatts_{-1.0};
  std::thread thread_;
};
